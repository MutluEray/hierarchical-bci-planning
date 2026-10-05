"""CAVIN-lite (DAY 3): two-level latent model for hierarchical planning, state-based (no pixels).

Latents
  c  effect code  : abstract "what changes" over a macro step of H micro-steps (a learned subgoal)
  z  motion code  : residual "how to do it" given the effect (path / timing / approach variations)

Generative model (what the planner uses)
  prior       p(c | s),  p(z | s, c)            state-conditioned Gaussians (learned to sample useful codes)
  meta-dynamics  s'' = s + f(s, c)               predicts the state after H micro-steps
  action generator  a_{1:H} = g(s, c, z)          decodes the effect into low-level actions (tanh-bounded)

Inference (training only), cascaded
  q_h(c | s, s'')           effect posterior sees the start and end state
  q_g(z | s, c, a_{1:H})    motion posterior sees the executed actions

NOTE on q_g: in my first scaffold q_g had no access to the actions, so z could not carry any motion
information. It now conditions on a_{1:H}. This is my reading of the cascaded inference of Fang et al.
(CoRL 2019); check the paper's exact conditioning before describing it as a faithful reproduction.
In the write-up call this "CAVIN-inspired / CAVIN-lite".

Loss (cascaded ELBO, Gaussian likelihoods with fixed variance)
  L = w_s * ||s'' - s''_hat||^2 + w_a * ||a - a_hat||^2
      + beta_c * KL(q_h || p(c|s)) + beta_z * KL(q_g || p(z|s,c))
with per-dimension free bits to help against posterior collapse.

All public methods take RAW states (metres); normalisation is internal.
"""
from __future__ import annotations

import torch
import torch.nn as nn

POS_SCALE = 1.0 / 0.3   # metres -> roughly unit range


def mlp(inp: int, out: int, hidden: int, layers: int = 2, layernorm: bool = True) -> nn.Sequential:
    mods, d = [], inp
    for _ in range(layers):
        mods.append(nn.Linear(d, hidden))
        if layernorm:
            mods.append(nn.LayerNorm(hidden))
        mods.append(nn.ReLU())
        d = hidden
    mods.append(nn.Linear(d, out))
    return nn.Sequential(*mods)


class CavinLite(nn.Module):
    def __init__(self, n_slots: int, horizon: int, action_dim: int = 3, effect_dim: int = 8,
                 motion_dim: int = 8, hidden: int = 256, state_dim: int | None = None):
        super().__init__()
        self.n_slots, self.H, self.ad = n_slots, horizon, action_dim
        self.sd = state_dim or 2 + 4 * n_slots          # [ee(2), obj xy(2n), active(n), held(n), (graspable(n))]
        self.cd, self.zd = effect_dim, motion_dim
        scale = torch.cat([torch.full((2 + 2 * n_slots,), POS_SCALE), torch.ones(self.sd - 2 - 2 * n_slots)])
        self.register_buffer("s_scale", scale)
        sd, cd, zd, H, ad = self.sd, self.cd, self.zd, self.H, self.ad
        # inference (training only)
        self.q_h = mlp(2 * sd, 2 * cd, hidden)
        self.q_g = mlp(sd + cd + H * ad, 2 * zd, hidden)
        # state-conditioned priors (used for planning)
        self.p_c = mlp(sd, 2 * cd, hidden)
        self.p_z = mlp(sd + cd, 2 * zd, hidden)
        # generative
        self.meta_dyn = mlp(sd + cd, sd, hidden)
        self.action_gen = mlp(sd + cd + zd, H * ad, hidden)

    # ------------------------------------------------------------------ helpers
    def norm(self, s: torch.Tensor) -> torch.Tensor:
        return s * self.s_scale

    def denorm(self, sn: torch.Tensor) -> torch.Tensor:
        return sn / self.s_scale

    @staticmethod
    def _gauss(x: torch.Tensor):
        mu, lv = x.chunk(2, dim=-1)
        return mu, lv.clamp(-6.0, 2.0)

    @staticmethod
    def _rsample(mu: torch.Tensor, lv: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * (0.5 * lv).exp()

    @staticmethod
    def _kl(mu_q, lv_q, mu_p, lv_p) -> torch.Tensor:
        """Elementwise KL(N(mu_q, e^lv_q) || N(mu_p, e^lv_p)), shape (B, d)."""
        return 0.5 * (lv_p - lv_q + (lv_q.exp() + (mu_q - mu_p) ** 2) / lv_p.exp() - 1.0)

    # ------------------------------------------------------------------ training
    def forward(self, s, actions, s_next, beta_c: float = 0.1, beta_z: float = 0.1, free_bits: float = 0.02,
                state_weight: float = 10.0, action_weight: float = 1.0):
        """s, s_next: (B, sd) raw; actions: (B, H, ad). Returns (loss, stats dict of detached tensors)."""
        B = s.shape[0]
        sn, s2n = self.norm(s), self.norm(s_next)
        a_flat = actions.reshape(B, -1)

        mu_qc, lv_qc = self._gauss(self.q_h(torch.cat([sn, s2n], -1)))
        c = self._rsample(mu_qc, lv_qc)
        mu_qz, lv_qz = self._gauss(self.q_g(torch.cat([sn, c, a_flat], -1)))
        z = self._rsample(mu_qz, lv_qz)

        mu_pc, lv_pc = self._gauss(self.p_c(sn))
        mu_pz, lv_pz = self._gauss(self.p_z(torch.cat([sn, c], -1)))

        delta = self.meta_dyn(torch.cat([sn, c], -1))
        a_pred = torch.tanh(self.action_gen(torch.cat([sn, c, z], -1))).view(B, self.H, self.ad)

        rec_s = ((sn + delta - s2n) ** 2).sum(-1).mean()
        rec_a = ((a_pred - actions) ** 2).sum((-1, -2)).mean()
        kl_c_dim = self._kl(mu_qc, lv_qc, mu_pc, lv_pc).mean(0)     # (cd,)
        kl_z_dim = self._kl(mu_qz, lv_qz, mu_pz, lv_pz).mean(0)     # (zd,)
        kl_c = torch.clamp(kl_c_dim, min=free_bits).sum()           # free bits per dimension
        kl_z = torch.clamp(kl_z_dim, min=free_bits).sum()

        loss = state_weight * rec_s + action_weight * rec_a + beta_c * kl_c + beta_z * kl_z
        stats = dict(loss=loss.detach(), rec_s=rec_s.detach(), rec_a=rec_a.detach(),
                     kl_c=kl_c_dim.sum().detach(), kl_z=kl_z_dim.sum().detach(),
                     kl_c_dim=kl_c_dim.detach(), kl_z_dim=kl_z_dim.detach())
        return loss, stats

    # ------------------------------------------------------------------ planning API (raw units)
    @torch.no_grad()
    def sample_effects(self, s: torch.Tensor, n: int) -> torch.Tensor:
        """s: (B, sd) -> effect codes (B, n, cd) sampled from the learned prior p(c | s)."""
        mu, lv = self._gauss(self.p_c(self.norm(s)))
        eps = torch.randn(s.shape[0], n, self.cd, device=s.device)
        return mu[:, None] + eps * (0.5 * lv).exp()[:, None]

    def _expand(self, s: torch.Tensor, c: torch.Tensor):
        squeeze = c.dim() == 2
        if squeeze:
            c = c[:, None]
        B, n, _ = c.shape
        sn = self.norm(s)[:, None].expand(B, n, self.sd).reshape(B * n, self.sd)
        return sn, c.reshape(B * n, -1), B, n, squeeze

    @torch.no_grad()
    def predict_next(self, s: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Meta-dynamics. s: (B, sd); c: (B, n, cd) or (B, cd). Returns raw next states (B, n, sd) or (B, sd)."""
        sn, c2, B, n, squeeze = self._expand(s, c)
        out = self.denorm(sn + self.meta_dyn(torch.cat([sn, c2], -1))).view(B, n, self.sd)
        return out[:, 0] if squeeze else out

    @torch.no_grad()
    def generate_actions(self, s: torch.Tensor, c: torch.Tensor, z: torch.Tensor | None = None,
                         deterministic: bool = False) -> torch.Tensor:
        """Action generator. Returns (B, n, H, ad) (or (B, H, ad) if c is 2-D). If z is None it is drawn
        from the prior p(z | s, c) (its mean if deterministic)."""
        sn, c2, B, n, squeeze = self._expand(s, c)
        if z is None:
            mu, lv = self._gauss(self.p_z(torch.cat([sn, c2], -1)))
            z2 = mu if deterministic else self._rsample(mu, lv)
        else:
            z2 = z.reshape(B * n, -1)
        a = torch.tanh(self.action_gen(torch.cat([sn, c2, z2], -1))).view(B, n, self.H, self.ad)
        return a[:, 0] if squeeze else a


def load_model(path: str, device: str = "cpu"):
    """Load a checkpoint written by train.train. Returns (model in eval mode, checkpoint dict)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    m = ck["meta"]
    model = CavinLite(m["n_slots"], m["H"], m["action_dim"], m["effect_dim"], m["motion_dim"], m["hidden"], m.get("state_dim"))
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck
