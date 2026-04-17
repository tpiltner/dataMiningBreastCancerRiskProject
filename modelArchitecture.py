# modelArchitecture.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from imageEncoder import get_image_encoder


# Monotone cumulative hazard head
class CumulativeProbabilityLayer(nn.Module):
    """
    Monotone cumulative logits:
      z_t = base + cumsum(softplus(inc_t))

    Produces nondecreasing logits across prediction horizons.
    """
    def __init__(self, in_dim: int, horizons: int = 5, dropout: float = 0.0):
        super().__init__()
        self.base = nn.Linear(in_dim, 1)
        self.inc = nn.Linear(in_dim, horizons)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        base = self.base(x)                 # [B,1]
        inc = F.softplus(self.inc(x))       # [B,H]
        z = base + torch.cumsum(inc, dim=1)
        return z


# View-attention pooling
class ViewAttentionPooling(nn.Module):
    """
    Learn attention weights over the 4 mammography views and produce a
    pooled exam-level vector.
    """
    def __init__(self, dim: int = 512, hidden: int = 128, temperature: float = 1.0):
        super().__init__()
        self.temperature = float(temperature)
        self.attn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

        # Initialize to approximately uniform attention
        nn.init.zeros_(self.attn[-1].weight)
        nn.init.zeros_(self.attn[-1].bias)

    @staticmethod
    def masked_softmax(logits: torch.Tensor, mask_keep: torch.Tensor) -> torch.Tensor:
        """
        logits: [B,V]
        mask_keep: [B,V] bool True=keep
        """
        mask_keep = mask_keep.bool()
        none_valid = (mask_keep.sum(dim=1) == 0)

        logits_f = logits.float().masked_fill(~mask_keep, float("-inf"))
        if none_valid.any():
            logits_f = torch.where(none_valid.unsqueeze(1), torch.zeros_like(logits_f), logits_f)

        weights = torch.softmax(logits_f, dim=1)
        if none_valid.any():
            weights = torch.where(none_valid.unsqueeze(1), torch.zeros_like(weights), weights)

        return weights.to(dtype=logits.dtype)

    def forward(self, v: torch.Tensor, mask_keep: torch.Tensor = None):
        """
        v: [B,4,C]
        mask_keep: [B,4] bool, optional

        returns:
          pooled: [B,C]
          weights: [B,4]
        """
        scores = self.attn(v).squeeze(-1)  # [B,4]
        if self.temperature != 1.0:
            scores = scores / self.temperature

        if mask_keep is None:
            weights = torch.softmax(scores.float(), dim=1).to(dtype=v.dtype)
        else:
            weights = self.masked_softmax(scores, mask_keep).to(dtype=v.dtype)

        pooled = (v * weights.unsqueeze(-1)).sum(dim=1)
        return pooled, weights


# Encoder wrapper
class ImageBackbone(nn.Module):
    """
    Shared ResNet-18 encoder.
    Encodes 4-view images into 4 view-level feature vectors.
    """
    def __init__(self, dim: int = 512, pretrained: bool = True, freeze_encoder: bool = False):
        super().__init__()
        self.dim = int(dim)
        self.freeze_encoder_flag = bool(freeze_encoder)

        self.encoder = get_image_encoder(pretrained=pretrained)

        if self.freeze_encoder_flag:
            for p in self.encoder.parameters():
                p.requires_grad = False
            if hasattr(self.encoder, "freeze"):
                self.encoder.freeze()

    def _set_bn_eval_if_frozen(self):
        if not self.freeze_encoder_flag:
            return
        for m in self.encoder.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def encode_views(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        imgs: [B,4,1,H,W]
        returns: [B,4,dim]
        """
        assert imgs.dim() == 5, f"Expected [B,4,1,H,W], got {tuple(imgs.shape)}"
        B, V, C, H, W = imgs.shape
        assert V == 4, f"Expected 4 views, got {V}"

        self._set_bn_eval_if_frozen()

        x = imgs.reshape(B * V, C, H, W)
        fmap = self.encoder(x, return_map=True)   # [B*4,dim,h,w]
        fmap = F.adaptive_avg_pool2d(fmap, (1, 1)).flatten(1)  # [B*4,dim]

        if fmap.shape[1] != self.dim:
            raise RuntimeError(f"Encoder returned dim {fmap.shape[1]}, expected {self.dim}")

        return fmap.view(B, V, self.dim)  # [B,4,dim]


# Baseline current-only model
class BaselineCurrentOnlyModel(nn.Module):
    """
    Baseline model architecture:
      4 current views
      -> encoder
      -> feature vector per view
      -> view attention pooling
      -> MLP head
      -> cumulative hazard prediction
    """
    def __init__(
        self,
        pretrained_encoder: bool = True,
        num_years: int = 5,
        dim: int = 512,
        mlp_hidden: int = 512,
        mlp_layers: int = 1,
        dropout: float = 0.2,
        freeze_encoder: bool = False,
        attn_hidden: int = 128,
        attn_temperature: float = 1.0,
        cum_dropout: float = 0.0,
    ):
        super().__init__()
        self.num_years = int(num_years)
        self.dim = int(dim)

        self.backbone = ImageBackbone(
            dim=self.dim,
            pretrained=pretrained_encoder,
            freeze_encoder=freeze_encoder,
        )
        self.view_pool = ViewAttentionPooling(
            dim=self.dim,
            hidden=attn_hidden,
            temperature=attn_temperature,
        )

        layers = []
        in_dim = self.dim
        L = max(1, int(mlp_layers))
        for i in range(L):
            out_dim = mlp_hidden if i < L - 1 else self.dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < L - 1:
                layers.append(nn.ReLU(inplace=True))
                if dropout and float(dropout) > 0:
                    layers.append(nn.Dropout(float(dropout)))
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)

        self.cum = CumulativeProbabilityLayer(
            self.dim,
            horizons=self.num_years,
            dropout=float(cum_dropout),
        )

    def forward(
        self,
        imgs: torch.Tensor,
        delta_feat: torch.Tensor = None,
        has_prior_views: torch.Tensor = None,
    ):
        """
        imgs: [B,4,1,H,W]

        Extra arguments are accepted for compatibility with older training code,
        but they are unused in the baseline model.
        """
        assert imgs.dim() == 5, f"Expected [B,4,1,H,W], got {tuple(imgs.shape)}"
        assert imgs.size(1) == 4, f"Expected 4 views, got {imgs.size(1)}"

        cur_vecs = self.backbone.encode_views(imgs)      # [B,4,dim]
        fused_vec, weights = self.view_pool(cur_vecs)    # [B,dim], [B,4]
        fused_vec = self.mlp(fused_vec)                  # [B,dim]
        logits = self.cum(fused_vec)                     # [B,num_years]

        return {
            "risk_prediction": {"pred_fused": logits},
            "attention_weights": weights,
        }