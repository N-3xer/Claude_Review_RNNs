"""
STAGE 2  -  paper sections 2-3: Backpropagation Through Time, truncation, vanishing/exploding gradients

This is the micrograd-style stage: we derive the backward pass BY HAND, then check it against
(a) autograd and (b) a numerical gradient. Squared-error loss, identity output (phi_o = identity):

    L = sum_t  l_t ,     l_t = 0.5 * ||O_t - Y_t||^2 / n                       (Eq. 5)

Backward recursion (read this slowly, it IS Eq. 6-12 written as a loop):
    dO_t   = (O_t - Y_t) / n
    dH_t   = dO_t W_ho^T  +  dA_{t+1} W_hh^T            <- from the output AND from the future
    dA_t   = dH_t * (1 - H_t^2)                         <- through tanh
    dW_xh += X_t^T dA_t        dW_hh += H_{t-1}^T dA_t        db_h += sum_n dA_t

Compare the last line with the paper's Eq. 11/12. Do they agree with this code? Decide for yourself (README, 'claims to test').
Everything here is float64 so the comparisons can be tight.
"""
import torch

from common import seed_all, header, check_close

torch.set_default_dtype(torch.float64)
NAMES = ["Wxh", "Whh", "bh", "Who", "bo"]


def loss_autograd(params, X, Y, trunc=None):
    Wxh, Whh, bh, Who, bo = params
    T, n, _ = X.shape
    H = X.new_zeros(n, Whh.shape[0])
    loss = 0.0
    for t in range(T):
        if trunc and t % trunc == 0 and t > 0:
            H = H.detach()                              # truncated BPTT: cut the graph at chunk boundaries
        H = torch.tanh(X[t] @ Wxh + H @ Whh + bh)       # (n,h)
        O = H @ Who + bo                                # (n,o)
        loss = loss + 0.5 * ((O - Y[t]) ** 2).sum() / n
    return loss


@torch.no_grad()
def backward_manual(params, X, Y, trunc=None):
    Wxh, Whh, bh, Who, bo = params
    T, n, d = X.shape
    h = Whh.shape[0]
    # ---- forward, caching what backward needs.  Hs[t+1] = H_t,  Hs[0] = H_{-1} = 0
    Hs = [X.new_zeros(n, h)]
    Os = []
    for t in range(T):
        Hs.append(torch.tanh(X[t] @ Wxh + Hs[t] @ Whh + bh))
        Os.append(Hs[t + 1] @ Who + bo)
    # ---- backward
    gWxh, gWhh, gbh, gWho, gbo = (torch.zeros_like(p) for p in params)
    dH_future = torch.zeros(n, h)
    for t in reversed(range(T)):
        dO = (Os[t] - Y[t]) / n                         # (n,o)
        gWho += Hs[t + 1].T @ dO                        # (h,n)@(n,o) -> (h,o)
        gbo += dO.sum(0)
        dH = dO @ Who.T + dH_future                     # (n,h)
        dA = dH * (1 - Hs[t + 1] ** 2)                  # (n,h)   tanh'(a) = 1 - tanh(a)^2
        gWxh += X[t].T @ dA                             # (d,n)@(n,h)
        gWhh += Hs[t].T @ dA                            # NOTE Hs[t] = H_{t-1}: the input of W_hh at step t
        gbh += dA.sum(0)
        dH_future = dA @ Whh.T                          # gradient arriving at H_{t-1}   (n,h)
        if trunc and t % trunc == 0:
            dH_future = torch.zeros_like(dH_future)     # first step of a chunk: do not go further back
    return gWxh, gWhh, gbh, gWho, gbo


def numeric_grad_entry(loss_fn, p, idx, eps=1e-6):
    """Central difference (f(x+eps) - f(x-eps)) / 2eps for ONE entry of p."""
    with torch.no_grad():
        old = p[idx].item()
        p[idx] = old + eps
        lp = loss_fn().item()
        p[idx] = old - eps
        lm = loss_fn().item()
        p[idx] = old
    return (lp - lm) / (2 * eps)


def gradient_vs_lag(rho, T=50, h=32, act="tanh", seed=0):
    """||dL/dH_k|| for every k, where L = sum(H_{T-1}) depends ONLY on the last state.
    W_hh = rho * (random orthogonal matrix): every singular value equals rho."""
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(h, h, generator=g))
    Whh = (rho * Q).requires_grad_()
    Wxh = torch.randn(4, h, generator=g) * 0.5
    X = torch.randn(T, 1, 4, generator=g)
    H = torch.zeros(1, h)
    Hs = []
    for t in range(T):
        pre = X[t] @ Wxh + H @ Whh
        H = torch.tanh(pre) if act == "tanh" else pre
        H.retain_grad()
        Hs.append(H)
    Hs[-1].sum().backward()
    return [Hs[k].grad.norm().item() for k in range(T)]


def clip_global_norm(grads, max_norm):
    total = torch.sqrt(sum((g**2).sum() for g in grads)).item()
    scale = min(1.0, max_norm / (total + 1e-12))
    return [g * scale for g in grads], total


if __name__ == "__main__":
    seed_all(0)
    T, n, d, h, o = 8, 3, 4, 5, 2
    X, Y = torch.randn(T, n, d), torch.randn(T, n, o)
    params = [torch.randn(d, h) * 0.5, torch.randn(h, h) * 0.5, torch.randn(h) * 0.1,
              torch.randn(h, o) * 0.5, torch.randn(o) * 0.1]
    for p in params:
        p.requires_grad_()

    header("1. manual BPTT  vs  autograd  (full and truncated)")
    for trunc in (None, 3):
        auto = torch.autograd.grad(loss_autograd(params, X, Y, trunc), params)
        manual = backward_manual(params, X, Y, trunc)
        for name, a, m_ in zip(NAMES, auto, manual):
            check_close(m_, a, f"{name:<4} trunc={trunc}", atol=1e-10, rtol=1e-8)

    header("2. third opinion: numerical gradient of ONE entry of W_hh")
    manual = backward_manual(params, X, Y)
    num = numeric_grad_entry(lambda: loss_autograd(params, X, Y), params[1], (1, 2))
    print(f"manual = {manual[1][1, 2].item():+.8f}   numeric = {num:+.8f}")
    assert abs(manual[1][1, 2].item() - num) < 1e-6

    header("3. vanishing / exploding gradients:  ||dL/dH_k||  at `lag` steps before the loss")
    lags = [0, 1, 2, 5, 10, 20, 40]
    for act in ("linear", "tanh"):
        print(f"\nactivation = {act}")
        print(f"{'rho':>6} | " + " ".join(f"{l:>9}" for l in lags))
        for rho in (0.8, 1.0, 1.2):
            norms = gradient_vs_lag(rho, T=50, act=act)
            row = [norms[49 - l] for l in lags]
            print(f"{rho:>6} | " + " ".join(f"{v:9.2e}" for v in row))
            if act == "linear":      # theory: Q orthogonal -> norm = sqrt(h) * rho^lag, exactly
                theory = torch.tensor([32**0.5 * rho**l for l in lags])
                check_close(torch.tensor(row), theory, f"linear, rho={rho}: sqrt(h)*rho^lag", atol=0, rtol=1e-8)

    header("4. gradient clipping (the standard fix for EXPLODING gradients)")
    grads = backward_manual(params, X, Y)
    clipped, total = clip_global_norm(grads, max_norm=0.1)
    new_total = torch.sqrt(sum((g**2).sum() for g in clipped)).item()
    print(f"global grad norm before = {total:.4f}, after clipping to 0.1 = {new_total:.4f}")
    assert new_total <= 0.1 + 1e-9

"""
EXERCISES  (hypothesis -> experiment -> result, in NOTES.md)
  1. In section 3's tanh table, rho=1.2 does NOT explode like the linear one. Hypothesis: tanh saturates and its
     derivative (1-H^2) shrinks. Test it: print mean(1 - H^2) per step for rho=1.2. Were you right?
  2. Find the largest rho for which the tanh gradient still survives 40 steps. Does it depend on the input scale (Wxh*0.5)?
  3. Delete the `(1 - Hs[t+1]**2)` factor from backward_manual. Which tests fail, by how much? Now you know
     what the paper's Eq. 11 is (not) accounting for.
  4. Change `Hs[t]` to `Hs[t + 1]` in the gWhh line. Which tests fail? What does that tell you about "which H does W_hh act on"?
  5. Truncation: for T=8, trunc=3, which entries of the manual gradient change vs full BPTT, and which don't? Why is gWho identical?
  6. Compute ||dL/dH_k|| in the tanh case for T=200 and plot log10(norm) vs lag. Fit a line. What is the slope?
"""
