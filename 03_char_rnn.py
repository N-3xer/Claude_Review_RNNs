"""
STAGE 3  -  makemore-style: a character-level language model with the RNN from scratch

    input  : characters  c_0 ... c_{T-1}      (one-hot  ==  row lookup in W_xh)
    target : the NEXT character  c_1 ... c_T
    loss   : cross-entropy on logits H_t W_ho + b_o

The corpus is generated from a tiny grammar, so we can compute the IRREDUCIBLE loss (the entropy of the generator).
Knowing the floor tells you how much of the gap is model error and how much is just randomness.

    python 03_char_rnn.py                 # default experiment
    python 03_char_rnn.py --text my.txt   # your own text (then there is no known floor)
"""
import argparse
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import seed_all, header, check_close

SUBJ = ["the cat", "the dog", "a bird", "the child", "my friend", "the robot"]
VERB = ["sees", "chases", "likes", "hears", "follows", "finds"]
OBJ = ["the ball", "a tree", "the moon", "a song", "the river", "a door"]
ADV = ["", " slowly", " today", " again", " quietly"]


def make_corpus(n_sent, seed=0):
    rng = random.Random(seed)
    sents = [f"{rng.choice(SUBJ)} {rng.choice(VERB)} {rng.choice(OBJ)}{rng.choice(ADV)}.\n" for _ in range(n_sent)]
    text = "".join(sents)
    nats_per_sentence = 3 * math.log(6) + math.log(5)       # 4 independent uniform choices
    return text, nats_per_sentence * n_sent / len(text)     # floor: nats per CHARACTER


def get_batch(data, n, T):
    ix = torch.randint(len(data) - T - 1, (n,))
    x = torch.stack([data[i:i + T] for i in ix])            # (n, T)
    y = torch.stack([data[i + 1:i + T + 1] for i in ix])    # (n, T)  same window shifted by one
    return x.T.contiguous(), y.T.contiguous()               # time-major (T, n)


class CharRNN(nn.Module):
    def __init__(self, V, h):
        super().__init__()
        self.V, self.h = V, h
        self.Wxh = nn.Parameter(torch.randn(V, h) / V**0.5)
        self.Whh = nn.Parameter(torch.randn(h, h) / h**0.5)
        self.bh = nn.Parameter(torch.zeros(h))
        self.Who = nn.Parameter(torch.randn(h, V) * 0.01)   # small -> initial logits ~ 0 -> initial loss ~ ln V
        self.bo = nn.Parameter(torch.zeros(V))

    def forward(self, x, H=None):                           # x: (T, n) integer ids
        T, n = x.shape
        if H is None:
            H = torch.zeros(n, self.h)
        XW = self.Wxh[x]                                    # (T, n, h)  == one_hot(x) @ Wxh   (checked in main)
        Hs = []
        for t in range(T):
            H = torch.tanh(XW[t] + H @ self.Whh + self.bh)  # (n, h)
            Hs.append(H)
        logits = torch.stack(Hs) @ self.Who + self.bo       # (T, n, V)
        return logits, H


def loss_fn(logits, y):                                     # (T,n,V), (T,n)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))


@torch.no_grad()
def sample(model, stoi, itos, start, n_chars=200, temp=1.0):
    ids = torch.tensor([[stoi[c]] for c in start])          # (len(start), 1)
    logits, H = model(ids)                                  # feed the prompt, keep the hidden state
    out = start
    for _ in range(n_chars):
        probs = F.softmax(logits[-1, 0] / temp, dim=-1)     # (V,)
        nxt = torch.multinomial(probs, 1).item()
        out += itos[nxt]
        logits, H = model(torch.tensor([[nxt]]), H)         # one step, carrying H
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--text", type=str, default=None)
    args = ap.parse_args()
    seed_all(0)

    if args.text:
        text, floor = open(args.text).read(), None
    else:
        text, floor = make_corpus(4000)
    chars = sorted(set(text))
    V = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = torch.tensor([stoi[c] for c in text])
    cut = int(0.9 * len(data))
    train, val = data[:cut], data[cut:]
    print(f"corpus: {len(text)} chars, vocab V={V}, ln(V)={math.log(V):.3f}"
          + (f", irreducible floor ~ {floor:.3f} nats/char" if floor else ""))

    header("sanity 1: one-hot @ W_xh  ==  row lookup W_xh[x]")
    m = CharRNN(V, args.hidden)
    x, y = get_batch(train, args.batch, args.T)
    check_close(F.one_hot(x, V).float() @ m.Wxh, m.Wxh[x], "one_hot(x) @ Wxh  vs  Wxh[x]")

    header("sanity 2: initial loss should be about ln(V)")
    with torch.no_grad():
        print(f"initial loss = {loss_fn(m(x)[0], y).item():.3f}   ln(V) = {math.log(V):.3f}")

    header("sanity 3: overfit ONE batch (the loss must fall far below ln V)")
    m2 = CharRNN(V, args.hidden)
    opt2 = torch.optim.Adam(m2.parameters(), lr=1e-2)
    for step in range(300):
        loss = loss_fn(m2(x)[0], y)
        opt2.zero_grad(); loss.backward(); opt2.step()
        if step % 100 == 0 or step == 299:
            print(f"  step {step:>3}  loss {loss.item():.4f}")

    header("training")
    model = CharRNN(V, args.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    @torch.no_grad()
    def evaluate(split, iters=20):
        ls = []
        for _ in range(iters):
            xb, yb = get_batch(split, args.batch, args.T)
            ls.append(loss_fn(model(xb)[0], yb).item())
        return sum(ls) / len(ls)

    for step in range(args.steps + 1):
        xb, yb = get_batch(train, args.batch, args.T)
        loss = loss_fn(model(xb)[0], yb)
        opt.zero_grad()
        loss.backward()
        gnorm = nn.utils.clip_grad_norm_(model.parameters(), args.clip)   # returns the norm BEFORE clipping
        opt.step()
        if step % 100 == 0:
            print(f"step {step:>5}  train {loss.item():.3f}  val {evaluate(val):.3f}  grad-norm {gnorm:.2f}")

    header("samples")
    for temp in (0.5, 1.0):
        print(f"--- temperature {temp}\n{sample(model, stoi, itos, text[:8], 200, temp)}\n")


if __name__ == "__main__":
    main()

"""
NOTE: every window starts from H = 0, so the first few characters of each window have no context. That is why the
loss stays ABOVE the floor even for a perfect model. Exercise 1 removes this.

EXERCISES
  1. Stateful training: split the data into n contiguous streams, carry H between batches (H = H.detach()).
     This is truncated BPTT exactly as in the paper. Does the loss get closer to the floor? Why?
  2. Remove clip_grad_norm_. Run 3 seeds. Does anything blow up? Log gnorm. At what hidden size / lr does it start to matter?
  3. Hypothesis: the model needs hidden >= some size to learn the grammar. Sweep --hidden 8,16,32,64,128 and plot final val loss.
  4. Change T from 32 to 8. What breaks? What does that say about how far back the model must look (longest dependency in the grammar)?
  5. Replace the hand-written forward with nn.RNN + nn.Linear + nn.Embedding. Same loss curve?
  6. Sampling at temperature -> 0 repeats one sentence. Why? What is the loss on its own samples?
"""
