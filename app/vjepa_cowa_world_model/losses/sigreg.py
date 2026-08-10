"""
SIGReg — Sketch Isotropic Gaussian Regularizer

Ported from le-wm (Maes et al., 2026).
Forces embedding distribution to be isotropic Gaussian via characteristic-function-based
statistic, preventing representation collapse without EMA or stop-gradient.

Reference: "LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels"
"""

import torch
import torch.nn as nn


class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer."""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        Args:
            proj: (T, B, D) — time-first embedding sequence
                  or (K, T, B, D) — multi-subspace embedding sequence
        Returns:
            scalar loss
        """
        if proj.dim() == 3:
            # original single-subspace path
            A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
            A = A.div_(A.norm(p=2, dim=0))
            x_t = (proj @ A).unsqueeze(-1) * self.t
            err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
            statistic = (err @ self.weights) * proj.size(-2)
            return statistic.mean()

        if proj.dim() == 4:
            # multi-subspace path: (K, T, B, D)
            A = torch.randn(proj.size(0), proj.size(-1), self.num_proj, device=proj.device)
            A = A.div_(A.norm(p=2, dim=1, keepdim=True))
            # x_t: (K, T, B, num_proj, knots)
            x_t = torch.einsum("ktbd,kdn->ktbn", proj, A).unsqueeze(-1) * self.t
            err = (x_t.cos().mean(dim=2) - self.phi).square() + x_t.sin().mean(dim=2).square()
            statistic = (err @ self.weights) * proj.size(2)
            return statistic.mean()

        raise ValueError(f"SIGReg expects 3D or 4D input, got shape {tuple(proj.shape)}")


class MultiSubspaceSIGReg(nn.Module):
    """Multi-subspace projection + SIGReg loss.

    init_mode:
        "orthogonal_frozen"     -> orthonormal init via QR, frozen buffer
        "random_frozen"         -> random Gaussian init, frozen buffer
        "random_trainable_soft" -> random init, trainable Parameter + soft orthogonality loss
    """

    def __init__(
        self,
        embed_dim=192,
        num_subspaces=4,
        subspace_dim=None,
        knots=17,
        num_proj=1024,
        init_mode="orthogonal_frozen",
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_subspaces = int(num_subspaces)
        if self.embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {self.embed_dim}")
        if self.num_subspaces <= 0:
            raise ValueError(f"num_subspaces must be positive, got {self.num_subspaces}")

        if subspace_dim is None:
            if self.embed_dim % self.num_subspaces != 0:
                raise ValueError(
                    "embed_dim must be divisible by num_subspaces when subspace_dim is not set. "
                    f"Got embed_dim={self.embed_dim}, num_subspaces={self.num_subspaces}."
                )
            self.subspace_dim = self.embed_dim // self.num_subspaces
        else:
            self.subspace_dim = int(subspace_dim)
            if self.subspace_dim <= 0:
                raise ValueError(f"subspace_dim must be positive, got {self.subspace_dim}")

        self.init_mode = str(init_mode)
        self.sigreg = SIGReg(knots=knots, num_proj=num_proj)

        if self.init_mode == "random_frozen":
            proj = self._init_random_projections(self.embed_dim, self.num_subspaces, self.subspace_dim)
            self.register_buffer("projection_matrices", proj)
        elif self.init_mode == "orthogonal_frozen":
            proj = self._init_orthogonal_projections(self.embed_dim, self.num_subspaces, self.subspace_dim)
            self.register_buffer("projection_matrices", proj)
        elif self.init_mode == "random_trainable_soft":
            proj = self._init_random_projections(self.embed_dim, self.num_subspaces, self.subspace_dim)
            self.projection_matrices = nn.Parameter(proj, requires_grad=True)
        else:
            raise ValueError(
                f"Unsupported init_mode ({init_mode}). "
                "Use 'random_frozen', 'orthogonal_frozen', or 'random_trainable_soft'."
            )

    @staticmethod
    def _init_random_projections(embed_dim, num_subspaces, subspace_dim):
        mats = []
        for _ in range(num_subspaces):
            mats.append(torch.randn(subspace_dim, embed_dim))
        return torch.stack(mats, dim=0)

    @staticmethod
    def _init_orthogonal_projections(embed_dim, num_subspaces, subspace_dim):
        mats = []
        for _ in range(num_subspaces):
            q, _ = torch.linalg.qr(torch.randn(embed_dim, subspace_dim), mode="reduced")
            mats.append(q.transpose(0, 1))
        return torch.stack(mats, dim=0)

    def get_projection_matrices(self):
        """Return stacked projection matrices with shape (K, d', D)."""
        return self.projection_matrices

    def project(self, emb):
        """
        Args:
            emb: (B, T, D)
        Returns:
            (K, T, B, d')
        """
        if emb.size(-1) != self.embed_dim:
            raise ValueError(f"Expected emb dim {self.embed_dim}, got {emb.size(-1)}")
        proj = torch.einsum("btd,ked->btke", emb, self.get_projection_matrices())
        return proj.permute(2, 1, 0, 3).contiguous()

    def forward(self, emb):
        """
        Args:
            emb: (B, T, D)
        Returns:
            scalar loss
        """
        subspace_emb = self.project(emb)
        return self.sigreg(subspace_emb)

    def orthogonality_loss(self):
        """Return mean ||A A^T - I||_F across subspaces.

        Returns a scalar zero tensor unless soft orthogonality mode is used.
        """
        if self.init_mode != "random_trainable_soft":
            return self.projection_matrices.new_zeros(())

        A = self.projection_matrices.float()  # (K, d', D)
        gram = A @ A.transpose(-1, -2)  # (K, d', d')
        eye = torch.eye(self.subspace_dim, device=A.device, dtype=A.dtype).unsqueeze(0)
        diff = gram - eye
        return torch.linalg.matrix_norm(diff, ord="fro", dim=(-2, -1)).mean()


def pool_tokens_per_frame(z_proj, tokens_per_frame):
    """Pool multi-token-per-frame projected embeddings for SIGReg.

    Args:
        z_proj: (B, T*P, embed_dim) — projected encoder output with P tokens per frame
        tokens_per_frame: int — number of tokens per video frame
    Returns:
        (T, B, embed_dim) — mean-pooled per-frame embeddings, time-first for SIGReg
    """
    B, TP, D = z_proj.shape
    T = TP // tokens_per_frame
    z_pooled = z_proj.reshape(B, T, tokens_per_frame, D).mean(dim=2)  # (B, T, D)
    return z_pooled.transpose(0, 1)  # (T, B, D)
