"""
STAGE 5  -  paper sections 5-6: deep RNNs (Eq. 19-21) and bidirectional RNNs (Eq. 22-24)

    deep  : H^(l)_t = phi( H^(l-1)_t , H^(l)_{t-1} )       layer l reads layer l-1's states as its input
    bidir : forward state reads X_0..X_t ;  backward state reads X_{T-1}..X_t ;  O_t uses BOTH

Implementation trick for the backward direction: flip time, run the SAME forward routine, flip back.
Then test the whole stack against torch.nn.RNN(num_layers=L, bidirectional=True).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import seed_all, header, check_shape, check_close


def rnn_layer(X, Wxh, Whh, bh):                       # X (T, n, d_in) -> (T, n, h)
    H = X.new_zeros(X.shape[1], Whh.shape[0])
    out = []
    for t in range(X.shape[0]):
        H = torch.tanh(X[t] @ Wxh + H @ Whh + bh)
        out.append(H)
    return torch.stack(out)


def init_layer(d_in, h):
    return dict(Wxh=torch.randn(d_in, h) / d_in**0.5, Whh=torch.randn(h, h) / h**0.5, bh=torch.randn(h) * 0.1)


def bi_layer(X, pf, pb):
    Hf = rnn_layer(X, **pf)                           # (T,n,h)   Hf[t] has seen X_0..X_t
    Hb = rnn_layer(X.flip(0), **pb).flip(0)           # (T,n,h)   run backwards, flip so index t lines up: seen X_t..X_{T-1}
    return torch.cat([Hf, Hb], dim=-1)                # (T,n,2h)  concat on the FEATURE axis (so W_ho is 2h x o)


def deep_bi_rnn(X, layers):                           # layers: list of (forward_params, backward_params)
    for pf, pb in layers:
        X = bi_layer(X, pf, pb)                       # next layer's input is 2h wide
    return X


def init_deep_bi(d, h, L):
    layers = []
    for l in range(L):
        d_in = d if l == 0 else 2 * h
        layers.append((init_layer(d_in, h), init_layer(d_in, h)))
    return layers


# --------------------------- the look-ahead experiment ---------------------------
class Tagger(nn.Module):
    def __init__(self, K, h, bidirectional):
        super().__init__()
        self.emb = nn.Embedding(K, 16)
        self.rnn = nn.RNN(16, h, bidirectional=bidirectional)
        self.head = nn.Linear(h * (2 if bidirectional else 1), K)

    def forward(self, x):                             # x (T, n) ids
        Hs, _ = self.rnn(self.emb(x))                 # (T, n, h) or (T, n, 2h)
        return self.head(Hs)                          # (T, n, K)


def run_lookahead(bidirectional, K=8, T=12, steps=300, seed=0):
    """Task: at every position t, output the NEXT symbol x_{t+1}.  Symbols are iid uniform, so a model that only
    sees x_0..x_t cannot do better than ln(K)."""
    seed_all(seed)
    model = Tagger(K, 32, bidirectional)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(steps):
        x = torch.randint(K, (T, 64))
        logits = model(x)
        loss = F.cross_entropy(logits[:-1].reshape(-1, K), x[1:].reshape(-1))   # drop the last position (no next symbol)
        opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()


if __name__ == "__main__":
    seed_all(0)
    T, n, d, h, L = 7, 3, 4, 5, 2
    X = torch.randn(T, n, d)
    layers = init_deep_bi(d, h, L)

    with torch.no_grad():
        header("1. shapes")
        out = deep_bi_rnn(X, layers)
        check_shape(out, (T, n, 2 * h), "top-layer [fwd ; bwd]")

        header("2. vs torch.nn.RNN(num_layers=2, bidirectional=True)")
        ref = nn.RNN(d, h, num_layers=L, bidirectional=True)
        for l, (pf, pb) in enumerate(layers):
            for suffix, p in (("", pf), ("_reverse", pb)):
                getattr(ref, f"weight_ih_l{l}{suffix}").copy_(p["Wxh"].T)
                getattr(ref, f"weight_hh_l{l}{suffix}").copy_(p["Whh"].T)
                getattr(ref, f"bias_ih_l{l}{suffix}").copy_(p["bh"])
                getattr(ref, f"bias_hh_l{l}{suffix}").zero_()
        check_close(out, ref(X)[0], "deep+bidirectional vs nn.RNN")

        header("3. what can each direction see?")
        base = bi_layer(X, layers[0][0], layers[0][1])
        X2 = X.clone(); X2[-1] += 1.0                 # poke the LAST input
        diff = (bi_layer(X2, layers[0][0], layers[0][1]) - base).abs()
        print("effect on forward half  at t=0..T-1:", diff[..., :h].amax(dim=(1, 2)))
        print("effect on backward half at t=0..T-1:", diff[..., h:].amax(dim=(1, 2)))

    header("4. experiment: predict the NEXT symbol (needs look-ahead).  chance = ln(8) = %.3f" % math.log(8))
    for bi in (False, True):
        print(f"bidirectional={bi!s:<5}  final loss = {run_lookahead(bi):.3f}")

"""
EXERCISES
  1. Paper Eq. 24 says the two states are concatenated 'on axis 0 (stacked on top of each other)', yet W_ho is 2h x o.
     Which axis does the code concatenate on, and why must it be that one? What shape would axis 0 give?
  2. Before running section 4, predict both losses. If bidirectional is NOT near zero, what could be wrong (lr? steps? size?)
  3. Why can't a bidirectional RNN be used to GENERATE text one token at a time? What does that imply about when BRNNs are allowed?
  4. Make the task 'predict x_{t+3}' for the bidirectional model. Harder? Why?
  5. Add dropout between layers in the deep stack. Where exactly should it go (hint: not on the recurrent connection)?
"""
