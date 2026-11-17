"""Cross-model latent space projectors and representation alignment metrics.

Supports:
- Orthogonal Procrustes (centered SVD alignment, rectangular d_src -> d_dst)
- Ridge Linear Regression (closed-form L2-regularized linear map)
- 2-layer MLP Projector (trained with AdamW + MSE + Cosine loss in PyTorch)
- Random unlearned baseline projector
- Linear CKA (Centered Kernel Alignment; Kornblith et al., 2019)
"""

from __future__ import annotations

import time
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ProjectorMethod = Literal["procrustes", "ridge", "mlp", "random"]


def centered_kernel_alignment(x: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor) -> float:
    """Compute Linear CKA between two representation matrices (N, d_x) and (N, d_y)."""
    x_t = torch.as_tensor(x, dtype=torch.float64)
    y_t = torch.as_tensor(y, dtype=torch.float64)
    if x_t.ndim != 2 or y_t.ndim != 2:
        raise ValueError(f"Expected 2D tensors, got {x_t.shape} and {y_t.shape}")
    if x_t.shape[0] != y_t.shape[0]:
        raise ValueError(f"Sample count mismatch: {x_t.shape[0]} vs {y_t.shape[0]}")

    x_c = x_t - x_t.mean(dim=0, keepdim=True)
    y_c = y_t - y_t.mean(dim=0, keepdim=True)

    cross = x_c.T @ y_c
    hsic_xy = torch.sum(cross * cross)
    hsic_xx = torch.sum((x_c.T @ x_c) ** 2)
    hsic_yy = torch.sum((y_c.T @ y_c) ** 2)

    denom = torch.sqrt(hsic_xx * hsic_yy)
    if denom.item() <= 1e-12:
        return 0.0
    return float((hsic_xy / denom).item())


def mse_loss(pred: np.ndarray | torch.Tensor, target: np.ndarray | torch.Tensor) -> float:
    p = torch.as_tensor(pred, dtype=torch.float32)
    t = torch.as_tensor(target, dtype=torch.float32)
    return float(F.mse_loss(p, t).item())


def cosine_similarity_mean(pred: np.ndarray | torch.Tensor, target: np.ndarray | torch.Tensor) -> float:
    p = torch.as_tensor(pred, dtype=torch.float32)
    t = torch.as_tensor(target, dtype=torch.float32)
    return float(F.cosine_similarity(p, t, dim=-1).mean().item())


class _MLPNet(nn.Module):
    def __init__(self, d_src: int, d_dst: int, hidden_dim: int = 1024, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_src, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_dst),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LatentProjector:
    """Projects hidden states from source model space R^{d_src} to target model space R^{d_dst}."""

    def __init__(
        self,
        d_src: int,
        d_dst: int,
        method: ProjectorMethod = "ridge",
        ridge_alpha: float = 10.0,
        mlp_hidden_dim: int = 1024,
        mlp_epochs: int = 300,
        mlp_lr: float = 1e-3,
        mlp_weight_decay: float = 1e-2,
        cosine_loss_weight: float = 0.5,
        seed: int = 42,
    ) -> None:
        self.d_src = d_src
        self.d_dst = d_dst
        self.method = method
        self.ridge_alpha = ridge_alpha
        self.mlp_hidden_dim = mlp_hidden_dim
        self.mlp_epochs = mlp_epochs
        self.mlp_lr = mlp_lr
        self.mlp_weight_decay = mlp_weight_decay
        self.cosine_loss_weight = cosine_loss_weight
        self.seed = seed

        self.w: torch.Tensor | None = None
        self.b: torch.Tensor | None = None
        self.scale: float = 1.0
        self.x_mean: torch.Tensor | None = None
        self.y_mean: torch.Tensor | None = None
        self.mlp: _MLPNet | None = None
        self.train_time_ms: float = 0.0

        if method == "random":
            g = torch.Generator().manual_seed(seed)
            w = torch.empty((d_src, d_dst), dtype=torch.float32)
            nn.init.orthogonal_(w, generator=g)
            self.w = w
            self.b = torch.zeros((d_dst,), dtype=torch.float32)

    def fit(self, x_src: np.ndarray | torch.Tensor, y_dst: np.ndarray | torch.Tensor) -> "LatentProjector":
        x = torch.as_tensor(x_src, dtype=torch.float32)
        y = torch.as_tensor(y_dst, dtype=torch.float32)
        if x.shape[1] != self.d_src or y.shape[1] != self.d_dst:
            raise ValueError(f"Dimension mismatch: got {x.shape}->{y.shape}, expected (*,{self.d_src})->(*,{self.d_dst})")

        t0 = time.perf_counter()

        if self.method == "random":
            self.train_time_ms = (time.perf_counter() - t0) * 1000.0
            return self

        if self.method == "ridge":
            x_mean = x.mean(dim=0, keepdim=True)
            y_mean = y.mean(dim=0, keepdim=True)
            xc = x - x_mean
            yc = y - y_mean

            xtx = xc.T @ xc + self.ridge_alpha * torch.eye(self.d_src, dtype=torch.float32)
            xty = xc.T @ yc
            self.w = torch.linalg.solve(xtx, xty)
            self.b = (y_mean - x_mean @ self.w).squeeze(0)

        elif self.method == "procrustes":
            x_mean = x.mean(dim=0, keepdim=True)
            y_mean = y.mean(dim=0, keepdim=True)
            xc = x - x_mean
            yc = y - y_mean

            m = xc.T @ yc  # (d_src, d_dst)
            u, s, vt = torch.linalg.svd(m, full_matrices=False)
            self.w = u @ vt  # (d_src, d_dst) with orthonormal rows/cols
            denom = torch.sum(xc * xc).clamp_min(1e-12)
            self.scale = float((torch.sum(s) / denom).item())
            self.x_mean = x_mean.squeeze(0)
            self.y_mean = y_mean.squeeze(0)

        elif self.method == "mlp":
            torch.manual_seed(self.seed)
            self.x_mean = x.mean(dim=0, keepdim=True)
            x_std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
            self._x_std = x_std
            xn = (x - self.x_mean) / x_std

            self.mlp = _MLPNet(self.d_src, self.d_dst, hidden_dim=self.mlp_hidden_dim)
            opt = torch.optim.AdamW(self.mlp.parameters(), lr=self.mlp_lr, weight_decay=self.mlp_weight_decay)
            self.mlp.train()
            for _ in range(self.mlp_epochs):
                opt.zero_grad()
                pred = self.mlp(xn)
                loss_mse = F.mse_loss(pred, y)
                loss_cos = 1.0 - F.cosine_similarity(pred, y, dim=-1).mean()
                loss = loss_mse + self.cosine_loss_weight * loss_cos
                loss.backward()
                opt.step()
            self.mlp.eval()
        else:
            raise ValueError(f"Unknown projector method: {self.method}")

        self.train_time_ms = (time.perf_counter() - t0) * 1000.0
        return self

    @torch.no_grad()
    def transform(self, x_src: np.ndarray | torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(x_src, dtype=torch.float32)
        if self.method in ("ridge", "random"):
            assert self.w is not None and self.b is not None
            return x @ self.w + self.b
        if self.method == "procrustes":
            assert self.w is not None and self.x_mean is not None and self.y_mean is not None
            return self.scale * ((x - self.x_mean) @ self.w) + self.y_mean
        if self.method == "mlp":
            assert self.mlp is not None and self.x_mean is not None
            xn = (x - self.x_mean) / self._x_std
            return self.mlp(xn)
        raise ValueError(f"Unknown projector method: {self.method}")
