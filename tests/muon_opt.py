"""Muon (Moonlight) - optimizador para el harness de smokes.

Implementacion del Muon con el ajuste de escala de Moonlight (Liu et al.
2025, arXiv:2502.16982): sobre matrices 2D del transformer se aplica
momentum Nesterov + ortogonalizacion Newton-Schulz (5 iters) con rescale
0.2*sqrt(max(A,B)), y decay aplicado en la actualizacion (W -= lr*(O +
lambda*W)). Embeddings, head de salida y parametros 1D/0D se optimizan
con AdamW (recomendacion del paper: las capas de entrada/salida NO van
por Muon).

Con el rescale de Moonlight se pueden reusar los mismos lr/wd que AdamW
sin re-tuning. Costo del Newton-Schulz en matrices chicas (<=128x128,
nuestro caso) es despreciable frente al paso.

Referencias: Keller Jordan et al. (Muon, 2024); Moonlight (2025) formula
(4): W_t = W_{t-1} - eta_t (0.2 * O_t * sqrt(max(A,B)) + lambda W_{t-1}).
"""

import torch
import torch.nn as nn


def zeropower_via_newtonschulz5(G, steps: int = 5, eps: float = 1e-7):
    """Ortogonalizacion Newton-Schulz de 5 pasos (coeficientes KJ)."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    if G.size(0) > G.size(1):
        X = X.T
    X = X / (X.norm() + eps)  # top singular value <= 1
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


def split_params(model):
    """muon_params: matrices 2D de nn.Linear (excepto el head final).
    adam_params: embeddings, head, biases y todo parametro 1D/0D."""
    muon, adam = [], []
    for mname, mod in model.named_modules():
        if isinstance(mod, nn.Embedding):
            adam.append(mod.weight)
        elif isinstance(mod, nn.Linear):
            if mname == 'head' or mname.endswith('.head'):
                adam.append(mod.weight)
            else:
                muon.append(mod.weight)
    covered = set(id(p) for p in (muon + adam))
    for p in model.parameters():
        if id(p) not in covered:
            adam.append(p)
    return muon, adam


class Muon(torch.optim.Optimizer):
    """Muon + AdamW combinados (el paso llama a ambos)."""

    def __init__(self, model, lr: float = 1e-3, momentum: float = 0.95,
                 weight_decay: float = 0.01, ns_steps: int = 5,
                 adam_betas=(0.9, 0.999), adam_eps: float = 1e-8):
        super().__init__([{'params': []}], dict(lr=lr, momentum=momentum,
                                                weight_decay=weight_decay))
        self.ns_steps = ns_steps
        self.muon_groups, adam_params = split_params(model)
        self.muon_mom = [torch.zeros_like(p) for p in self.muon_groups]
        fused = bool(adam_params) and all(p.is_cuda for p in adam_params)
        self.adam = torch.optim.AdamW(adam_params, lr=lr,
                                      weight_decay=weight_decay,
                                      betas=adam_betas, eps=adam_eps,
                                      fused=fused)

    @torch.no_grad()
    def step(self, closure=None):
        lr = self.defaults['lr']
        momentum = self.defaults['momentum']
        wd = self.defaults['weight_decay']
        for p, buf in zip(self.muon_groups, self.muon_mom):
            if p.grad is None:
                continue
            g = p.grad
            buf.mul_(momentum).add_(g)
            g = buf + momentum * g                  # Nesterov
            O = zeropower_via_newtonschulz5(g, self.ns_steps)
            O.mul_(0.2 * (max(p.shape) ** 0.5))     # Moonlight rescale
            p.add_(O, alpha=-lr)
            p.mul_(1.0 - wd * lr)                   # weight decay
        self.adam.step()

    def zero_grad(self, set_to_none: bool = True):
        for p in self.muon_groups:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()
        self.adam.zero_grad(set_to_none=set_to_none)


def make_optimizer(model, name: str = 'adamw', lr: float = 1e-3,
                   weight_decay: float = 0.01):
    if name == 'adamw':
        fused = all(p.is_cuda for p in model.parameters())
        return torch.optim.AdamW(model.parameters(), lr=lr,
                                 weight_decay=weight_decay, fused=fused)
    if name == 'muon':
        return Muon(model, lr=lr, weight_decay=weight_decay)
    raise ValueError(f'optimizador desconocido: {name}')
