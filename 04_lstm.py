"""
STAGE 4  -  paper section 4: LSTM (Eq. 13-18), then the experiment the paper only talks about

    O_t = sigmoid(X_t W_xo + H_{t-1} W_ho + b_o)         output gate      (13)
    I_t = sigmoid(X_t W_xi + H_{t-1} W_hi + b_i)         input gate       (14)
    F_t = sigmoid(X_t W_xf + H_{t-1} W_hf + b_f)         forget gate      (15)
    C~_t = tanh  (X_t W_xc + H_{t-1} W_hc + b_c)         candidate cell   (16)
    C_t = F_t * C_{t-1} + I_t * C~_t                                      (17)
    H_t = O_t * tanh(C_t)                                                 (18)

(Notation clash in the paper: the output GATE matrix W_ho is (h x h); the RNN READOUT matrix W_ho of stage 1 is (h x o).)

Engineering essential shown here: four small matmuls == ONE big matmul + chunk.  Every real implementation does this.
"""
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import seed_all, header, check_shape, check_close


class LSTM(nn.Module):
    def __init__(self, d, h, forget_bias=1.0):
        super().__init__()
        self.h = h
        self.Wx = nn.Parameter(torch.randn(d, 4 * h) / d**0.5)    # columns: [ input | forget | candidate | output ]
        self.Wh = nn.Parameter(torch.randn(h, 4 * h) / h**0.5)
        b = torch.zeros(4 * h)
        b[h:2 * h] = forget_bias                                  # start by REMEMBERING (try 0.0 in the experiment)
        self.b = nn.Parameter(b)

    def step(self, x, H, C):                                      # x (n,d)   H, C (n,h)
        gates = x @ self.Wx + H @ self.Wh + self.b                # (n, 4h)
        i_g, f_g, c_t, o_g = gates.chunk(4, dim=1)                # 4 x (n, h)
        i_g, f_g, o_g = torch.sigmoid(i_g), torch.sigmoid(f_g), torch.sigmoid(o_g)
        c_t = torch.tanh(c_t)
        C = f_g * C + i_g * c_t                                   # (17)  elementwise: the cell is a *conveyor belt*
        H = o_g * torch.tanh(C)                                   # (18)
        return H, C

    def forward(self, X, state=None):                             # X: (T, n, d)
        T, n, _ = X.shape
        H, C = state if state is not None else (X.new_zeros(n, self.h), X.new_zeros(n, self.h))
        Hs = []
        for t in range(T):
            H, C = self.step(X[t], H, C)
            Hs.append(H)
        return torch.stack(Hs), (H, C)                            # (T,n,h), final state


def lstm_step_paper(x, H, C, p):
    """Literal transcription of Eq. 13-18 with 12 separate parameters."""
    O = torch.sigmoid(x @ p["Wxo"] + H @ p["Who"] + p["bo"])     # (13)
    I = torch.sigmoid(x @ p["Wxi"] + H @ p["Whi"] + p["bi"])     # (14)
    Fg = torch.sigmoid(x @ p["Wxf"] + H @ p["Whf"] + p["bf"])    # (15)
    C_tilde = torch.tanh(x @ p["Wxc"] + H @ p["Whc"] + p["bc"])  # (16)
    C_new = Fg * C + I * C_tilde                                  # (17)
    return O * torch.tanh(C_new), C_new                           # (18)


def paper_params(m):
    h = m.h
    sl = {"i": slice(0, h), "f": slice(h, 2 * h), "c": slice(2 * h, 3 * h), "o": slice(3 * h, 4 * h)}
    p = {}
    for g, s in sl.items():
        p[f"Wx{g}"], p[f"Wh{g}"], p[f"b{g}"] = m.Wx[:, s], m.Wh[:, s], m.b[s]
    return p


# ------------------------- the long-range-memory experiment -------------------------
class TanhRNN(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.Wxh = nn.Parameter(torch.randn(d, h) / d**0.5)
        self.Whh = nn.Parameter(torch.randn(h, h) / h**0.5)
        self.bh = nn.Parameter(torch.zeros(h))

    def forward(self, X):
        H = X.new_zeros(X.shape[1], self.h)
        Hs = []
        for t in range(X.shape[0]):
            H = torch.tanh(X[t] @ self.Wxh + H @ self.Whh + self.bh)
            Hs.append(H)
        return torch.stack(Hs), H


def make_task(n, T, K=4):
    """Remember which of K classes appeared at t=0. Every step also carries distractor noise."""
    y = torch.randint(K, (n,))
    X = torch.zeros(T, n, K + 1)
    X[0, torch.arange(n), y] = 1.0           # the one thing to remember (first K channels, only at t=0)
    X[:, :, K] = torch.randn(T, n)           # distractor channel, every step
    return X, y


class Classifier(nn.Module):
    def __init__(self, kind, K, h, forget_bias):
        super().__init__()
        self.rnn = TanhRNN(K + 1, h) if kind == "rnn" else LSTM(K + 1, h, forget_bias)
        self.head = nn.Linear(h, K)

    def forward(self, X):
        Hs, _ = self.rnn(X)
        return self.head(Hs[-1])             # read out ONLY at the last step -> loss must reach back T steps


def train_eval(kind, T, steps, h, lr, forget_bias, seed=0, K=4):
    seed_all(seed)
    model = Classifier(kind, K, h, forget_bias)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        X, y = make_task(64, T, K)
        loss = F.cross_entropy(model(X), y)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    with torch.no_grad():
        Xv, yv = make_task(1000, T, K)
        return (model(Xv).argmax(-1) == yv).float().mean().item()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--forget_bias", type=float, default=1.0)
    ap.add_argument("--lags", type=int, nargs="+", default=[5, 20, 50])
    args = ap.parse_args()

    seed_all(0)
    d, h, T, n = 5, 8, 6, 4
    X = torch.randn(T, n, d)
    m = LSTM(d, h)
    with torch.no_grad():
        nn.init.normal_(m.b, std=0.3)                            # random biases so the tests are not trivial

        header("1. shapes")
        Hs, (Hn, Cn) = m(X)
        check_shape(Hs, (T, n, h), "H (all steps)")
        check_shape(Cn, (n, h), "C (final)")

        header("2. one big matmul  ==  the paper's 12 parameters")
        H0, C0 = torch.randn(n, h), torch.randn(n, h)
        Ha, Ca = m.step(X[0], H0, C0)
        Hb, Cb = lstm_step_paper(X[0], H0, C0, paper_params(m))
        check_close(Ha, Hb, "H  fused vs paper")
        check_close(Ca, Cb, "C  fused vs paper")

        header("3. vs torch.nn.LSTM (gate order in PyTorch is i, f, g, o - same as mine)")
        ref = nn.LSTM(d, h)
        ref.weight_ih_l0.copy_(m.Wx.T)
        ref.weight_hh_l0.copy_(m.Wh.T)
        ref.bias_ih_l0.copy_(m.b)
        ref.bias_hh_l0.zero_()
        out_ref, (h_n, c_n) = ref(X)
        check_close(Hs, out_ref, "H  all steps vs nn.LSTM")
        check_close(Cn, c_n[0], "C  final vs nn.LSTM")

    header("4. experiment: remember the first symbol across `lag` noisy steps")
    chance = 0.25
    print(f"chance accuracy = {chance:.2f}.  hidden={args.hidden}, steps={args.steps}, lr={args.lr}, forget_bias={args.forget_bias}")
    print(f"{'lag':>5} | {'tanh-RNN':>9} {'LSTM':>9}")
    for T_ in args.lags:
        r = train_eval("rnn", T_, args.steps, args.hidden, args.lr, args.forget_bias)
        l = train_eval("lstm", T_, args.steps, args.hidden, args.lr, args.forget_bias)
        print(f"{T_:>5} | {r:>9.2f} {l:>9.2f}")

"""
EXERCISES  (this stage is mostly about running experiments honestly; I have NOT tuned the defaults)
  1. Before running: write your prediction for the 3x2 table. After: were you right? If the LSTM also fails at lag 50,
     do NOT conclude 'LSTMs do not work'. List 3 things that could explain it and test each one.
  2. Run with --forget_bias 0.0. Compare with 1.0 at lag 50. Hypothesis for why it matters (look at Eq. 17)?
  3. Reuse gradient_vs_lag from stage 2's idea: measure ||dL/dC_k|| through the LSTM cell state for lag 50. Where is the
     gradient 'highway'? (Hint: with F=1 and I=0, what is dC_t/dC_{t-1}?)
  4. Implement a GRU (2 gates: reset, update) from the d2l chapter. Compare parameter counts and the table above.
  5. Replace the fused matmul with 4 separate nn.Linear layers. Time 200 forward passes of both (time.perf_counter).
"""
