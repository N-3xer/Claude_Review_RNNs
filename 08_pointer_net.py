"""
STAGE 8  -  paper section 9: Pointer Networks

Idea: the output "vocabulary" is the set of INPUT POSITIONS.  No blending of encoder states into a context vector:
the attention distribution itself is the output (Eq. 29):

        p(Y_t = t' | ...) = softmax_t'( v^T tanh( W1 E_t' + W2 S_t ) )

Task: SORT.  Input: T random numbers in [0,1].  Output: the indices that list them in ascending order.
Output vocabulary size == T, and it changes with the input length - impossible for a fixed softmax over words.
"""
import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import seed_all, header


class PtrNet(nn.Module):
    def __init__(self, h=64, a=64):
        super().__init__()
        self.enc = nn.LSTM(1, h)                       # (T, n, 1) -> (T, n, h)
        self.dec = nn.LSTMCell(1, h)                   # input: the VALUE it just pointed at
        self.W1 = nn.Linear(h, a, bias=False)          # encoder side of the score
        self.W2 = nn.Linear(h, a, bias=True)           # decoder side of the score
        self.v = nn.Linear(a, 1, bias=False)
        self.start = nn.Parameter(torch.zeros(1))      # learned "GO" input for step 0

    def forward(self, x, idx=None):                    # x (T, n, 1);  idx (T, n) targets for teacher forcing, or None
        T, n, _ = x.shape
        E, (h_n, c_n) = self.enc(x)                    # E (T, n, h)
        h, c = h_n[0], c_n[0]                          # (n, h)
        keyed = self.W1(E)                             # (T, n, a)   computed once
        inp = self.start.expand(n, 1)                  # (n, 1)
        ar = torch.arange(n)
        out = []
        for j in range(T):
            h, c = self.dec(inp, (h, c))
            u = self.v(torch.tanh(keyed + self.W2(h))).squeeze(-1)   # (T, n): one score per INPUT POSITION
            out.append(u.T)                            # (n, T_in)
            pick = idx[j] if idx is not None else u.argmax(0)        # (n,) the position pointed at
            inp = x[pick, ar]                          # (n, 1)  look up the value at that position
        return torch.stack(out)                        # (T_out, n, T_in)  logits over input positions


def make_batch(n, T):
    x = torch.rand(T, n, 1)
    idx = x.squeeze(-1).argsort(dim=0)                 # (T, n): idx[j, b] = position of the j-th smallest number in column b
    return x, idx


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=6)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()
    seed_all(0)
    T = args.T

    model = PtrNet()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    header("sanity: untrained loss should be about ln(T) = %.3f" % math.log(T))
    with torch.no_grad():
        x, idx = make_batch(args.batch, T)
        print(f"initial loss = {F.cross_entropy(model(x, idx).reshape(-1, T), idx.reshape(-1)).item():.3f}")

    header("training")
    for step in range(args.steps + 1):
        x, idx = make_batch(args.batch, T)
        loss = F.cross_entropy(model(x, idx).reshape(-1, T), idx.reshape(-1))
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 500 == 0:
            print(f"step {step:>5}  loss {loss.item():.3f}")

    header("evaluation (greedy decoding, no teacher forcing)")
    with torch.no_grad():
        x, idx = make_batch(2000, T)
        pred = model(x).argmax(-1)                     # (T, n)
        print(f"per-position accuracy : {(pred == idx).float().mean().item():.3f}")
        print(f"whole-sequence accuracy: {(pred == idx).all(0).float().mean().item():.3f}   (chance = 1/{T}! = {1 / math.factorial(T):.4f})")
        v = x[:, 0, 0]
        print("\nexample input :", [round(t, 2) for t in v.tolist()])
        print("predicted     :", pred[:, 0].tolist(), "-> values", [round(v[i].item(), 2) for i in pred[:, 0]])
        print("true order    :", idx[:, 0].tolist())

"""
EXERCISES
  1. Train with T=6, then evaluate on T=10 without retraining (the model is length-agnostic!). What happens? Why is that
     impossible for a seq2seq with a fixed output softmax?
  2. The decoder may point at the same position twice. Add a mask (set already-chosen positions to -inf) at decoding time.
     How much does whole-sequence accuracy change? Is that a model fix or a decoding fix?
  3. Paper: 'the output only responds to the positions but not the input content'. Hmm - but what DOES the decoder feed back as input?
     Does the claim hold for this implementation?
  4. Replace the additive score by a dot-product score. Compare after 3000 steps; which trains faster?
  5. Change the task to 'convex hull of 2D points' (input dim 2). What must change in the code - and what does not?
"""
