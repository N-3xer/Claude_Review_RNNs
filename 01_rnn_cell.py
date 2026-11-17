"""
STAGE 1  -  paper section 1: the recurrent cell (Eq. 1-4)

    H_t = tanh( X_t W_xh + H_{t-1} W_hh + b_h )          (1)
    O_t = H_t W_ho + b_o                                  (2)   (phi_o = identity -> logits)

LAYOUT: time-major.   X: (T, n, d)   T = time steps, n = batch ("samples"), d = features.
This matches the paper, d2l, and nn.RNN's default.  From stage 7 on we switch to (B, T, D).

HOW TO USE THIS FILE
  1. Read the RNN class. Cover it. Write your own from the equations, shape comment on every line.
  2. Run:  python 01_rnn_cell.py   -> every [ok] line is a test you must be able to explain.
  3. Do the EXERCISES at the bottom. Predict first, write the prediction down, then run.
"""
import torch
import torch.nn as nn

from common import seed_all, header, check_shape, check_close


class RNN(nn.Module):
    def __init__(self, d, h, o):
        super().__init__()
        self.h = h
        self.Wxh = nn.Parameter(torch.randn(d, h) / d**0.5)   # (d, h)
        self.Whh = nn.Parameter(torch.randn(h, h) / h**0.5)   # (h, h)   ~unit spectral radius at init
        self.bh = nn.Parameter(torch.zeros(h))                # (h,)
        self.Who = nn.Parameter(torch.randn(h, o) * 0.01)     # (h, o)   small -> near-uniform logits at init
        self.bo = nn.Parameter(torch.zeros(o))                # (o,)

    def forward(self, X, H=None):                 # X: (T, n, d)
        T, n, d = X.shape
        if H is None:
            H = X.new_zeros(n, self.h)            # H_{-1} = 0
        Hs = []
        for t in range(T):                        # <- the loop over time CANNOT be removed: H_t needs H_{t-1}
            H = torch.tanh(X[t] @ self.Wxh + H @ self.Whh + self.bh)   # (n,d)@(d,h) + (n,h)@(h,h) + (h,) -> (n,h)
            Hs.append(H)
        Hs = torch.stack(Hs)                      # (T, n, h)
        O = Hs @ self.Who + self.bo               # (T,n,h)@(h,o) -> (T,n,o): ONE matmul for all time steps
        return O, Hs

    def forward_hoisted(self, X, H=None):
        """Same function. Which part of Eq. 1 depends on the recurrence, and which doesn't?"""
        T, n, d = X.shape
        if H is None:
            H = X.new_zeros(n, self.h)
        XW = X @ self.Wxh + self.bh               # (T,n,d)@(d,h) -> (T,n,h): input part for ALL t at once
        Hs = []
        for t in range(T):
            H = torch.tanh(XW[t] + H @ self.Whh)  # only this matmul is inherently sequential
            Hs.append(H)
        Hs = torch.stack(Hs)
        return Hs @ self.Who + self.bo, Hs


if __name__ == "__main__":
    seed_all(0)
    T, n, d, h, o = 6, 4, 5, 8, 3
    X = torch.randn(T, n, d)
    m = RNN(d, h, o)

    with torch.no_grad():
        header("1. shapes (predict them BEFORE you run)")
        O, Hs = m(X)
        check_shape(Hs, (T, n, h), "H  (all steps)")
        check_shape(O, (T, n, o), "O  (all steps)")

        header("2. does my RNN equal torch.nn.RNN?")
        ref = nn.RNN(d, h, nonlinearity="tanh")       # PyTorch stores weights as (out, in) = TRANSPOSE of the paper
        ref.weight_ih_l0.copy_(m.Wxh.T)               # (h, d)
        ref.weight_hh_l0.copy_(m.Whh.T)               # (h, h)
        ref.bias_ih_l0.copy_(m.bh)
        ref.bias_hh_l0.zero_()                        # PyTorch has two bias vectors; the paper has one
        Hs_ref, _ = ref(X)
        check_close(Hs, Hs_ref, "H  vs nn.RNN")

        header("3. hoisting X @ W_xh out of the loop")
        O2, Hs2 = m.forward_hoisted(X)
        check_close(Hs, Hs2, "H  loop vs hoisted")

        header("4. W_hh = 0  ->  the RNN collapses to a feedforward net (Eq. 3)")
        m0 = RNN(d, h, o)
        m0.load_state_dict(m.state_dict())
        m0.Whh.zero_()
        _, H0s = m0(X)
        check_close(H0s, torch.tanh(X @ m.Wxh + m.bh), "H(W_hh=0) vs tanh(X W_xh + b)")

        header("5. information only flows FORWARD in time")
        X2 = X.clone(); X2[0] += 1.0                  # poke the first input
        X3 = X.clone(); X3[-1] += 1.0                 # poke the last input
        _, Hs_a = m(X2)
        _, Hs_b = m(X3)
        print("|change in H_t| after poking X_0 , t=0..T-1:", (Hs_a - Hs).abs().amax(dim=(1, 2)))
        print("|change in H_t| after poking X_T-1, t=0..T-1:", (Hs_b - Hs).abs().amax(dim=(1, 2)))
        print("^ why is the second vector zero everywhere except the last entry? What does that mean for"
              " 'filling in a gap' tasks? (Stage 5)")

"""
EXERCISES  (write a 1-line prediction in NOTES.md first)
  1. Rewrite the loop body with torch.einsum. Same result?
  2. Make Whh = 3 * torch.eye(h). Run T=50 on random input. What happens to H? Why? (Stage 2 explains it.)
  3. Replace tanh with ReLU. What happens to the size of H over a long sequence? Why is tanh 'safe' here?
  4. Count parameters of RNN(d, h, o) by hand, then check with sum(p.numel() for p in m.parameters()).
  5. Make X a (n, T, d) tensor (batch-first). Change ONLY the indexing so forward still works. Which is
     cheaper per step, X[t] or X[:, t]? (hint: .is_contiguous())
"""
