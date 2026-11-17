"""
STAGE 6  -  paper section 7 + 8.1 + Appendix B: seq2seq and (Bahdanau/additive) attention

Task: REVERSE a digit sequence.  x = [3 1 4 1 5]  ->  y = [5 1 4 1 3].
  * tiny, synthetic, fully checkable
  * the correct alignment is the anti-diagonal, so we know exactly what a good attention matrix looks like

Two models, same everything except the context:
  no attention : decoder only gets the encoder summary vector (the bottleneck the paper describes)
  attention    : at every output step t the decoder computes weights over ALL encoder states
        e_{t,t'}   = v^T tanh( W_s S_{t-1} + W_h H_t' )          score  (Table 1 'Additive', concat form split in two)
        alpha_t,.  = softmax over t'                              (Eq. 28)
        C_t        = sum_t' alpha_{t,t'} H_t'                     (Eq. 27)
        S_t        = GRU( [emb(Y_{t-1}) ; C_t] , S_{t-1} )        (Eq. 26)
Time-major again: x (T, n).
"""
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import seed_all, header, check_close

D = 10                  # digits 0..9
BOS = D                 # start-of-sequence id for the decoder input (so the target vocab is D, the input vocab D+1)


def make_batch(n, T):
    x = torch.randint(D, (T, n))
    return x, x.flip(0)                                       # (T, n), (T, n)


class Seq2Seq(nn.Module):
    def __init__(self, h=64, e=16, attention=True):
        super().__init__()
        self.h, self.attention = h, attention
        self.emb_src = nn.Embedding(D, e)
        self.emb_tgt = nn.Embedding(D + 1, e)
        self.enc = nn.GRU(e, h, bidirectional=True)           # H_t' = [fwd ; bwd]   (Eq. 25)   -> (T, n, 2h)
        self.init_s = nn.Linear(2 * h, h)                     # encoder summary -> S_0
        self.dec = nn.GRUCell(e + (2 * h if attention else 0), h)
        self.W_s = nn.Linear(h, h, bias=False)
        self.W_h = nn.Linear(2 * h, h, bias=True)
        self.v = nn.Linear(h, 1, bias=False)
        self.out = nn.Linear(h + (2 * h if attention else 0), D)

    def forward(self, x, y=None):
        """y given -> teacher forcing (feed the TRUE previous token). y=None -> greedy decoding (feed own prediction)."""
        T, n = x.shape
        Hs, h_n = self.enc(self.emb_src(x))                   # Hs (T,n,2h) ; h_n (2,n,h) = [last fwd ; last bwd]
        summary = h_n.permute(1, 0, 2).reshape(n, 2 * self.h) # (n, 2h)
        S = torch.tanh(self.init_s(summary))                  # (n, h)
        prev = torch.full((n,), BOS, dtype=torch.long)
        keyed = self.W_h(Hs)                                  # (T,n,h)  the encoder half of the score, computed ONCE
        logits, alphas = [], []
        for t in range(T):
            inp = self.emb_tgt(prev)                          # (n, e)
            if self.attention:
                scores = self.v(torch.tanh(keyed + self.W_s(S))).squeeze(-1)   # (T,n)+(n,h) broadcast -> (T,n)
                a = F.softmax(scores, dim=0)                  # (T, n)   softmax over SOURCE positions, sums to 1
                C = (a.unsqueeze(-1) * Hs).sum(0)             # (T,n,1)*(T,n,2h) -> sum over T -> (n, 2h)
                alphas.append(a)
                S = self.dec(torch.cat([inp, C], dim=-1), S)  # (n,h)
                logits.append(self.out(torch.cat([S, C], dim=-1)))
            else:
                S = self.dec(inp, S)
                logits.append(self.out(S))
            prev = y[t] if y is not None else logits[-1].argmax(-1)
        return torch.stack(logits), (torch.stack(alphas) if alphas else None)   # (T,n,D), (T_out,T_in,n)


def check_additive_equivalence():
    """W_a [S ; H]  ==  W_s S + W_h H   -> lets us precompute the H part once instead of T times."""
    T, n, h = 6, 3, 8
    S, H = torch.randn(n, h), torch.randn(T, n, 2 * h)
    Wa = nn.Linear(3 * h, h)
    cat_form = Wa(torch.cat([S.expand(T, n, h), H], dim=-1))
    split_form = S @ Wa.weight[:, :h].T + H @ Wa.weight[:, h:].T + Wa.bias
    check_close(cat_form, split_form, "W_a[S;H]  vs  W_s S + W_h H")


def train(attention, args):
    seed_all(0)
    model = Seq2Seq(args.hidden, attention=attention)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    for step in range(args.steps + 1):
        x, y = make_batch(args.batch, args.T)
        logits, _ = model(x, y)
        loss = F.cross_entropy(logits.reshape(-1, D), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 250 == 0:
            print(f"  [{'attn' if attention else 'no-attn'}] step {step:>5}  loss {loss.item():.3f}")
    return model


@torch.no_grad()
def evaluate(model, args):
    x, y = make_batch(2000, args.T)
    logits, alphas = model(x)                                 # greedy decoding, no teacher forcing
    pred = logits.argmax(-1)
    return (pred == y).float().mean().item(), (pred == y).all(0).float().mean().item(), alphas


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=3e-3)
    args = ap.parse_args()

    header("check: concatenation form of the additive score == split form")
    check_additive_equivalence()

    header(f"training, T={args.T}  (chance per-token accuracy = 0.10)")
    results = {}
    for attn in (False, True):
        model = train(attn, args)
        tok, seq, alphas = evaluate(model, args)
        results[attn] = (model, tok, seq, alphas)

    header("results (greedy decoding)")
    for attn, (model, tok, seq, alphas) in results.items():
        print(f"{'attention' if attn else 'no attention':<14} token acc {tok:.3f}   whole-sequence acc {seq:.3f}")

    alphas = results[True][3]                                 # (T_out, T_in, n)
    print("\nattention argmax per output step (example 0):", alphas[:, :, 0].argmax(1).tolist())
    print("a perfect reverser would show              :", list(range(args.T - 1, -1, -1)))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.imshow(alphas[:, :, 0].numpy(), cmap="gray")
        plt.xlabel("source position t'"); plt.ylabel("output step t"); plt.title("alignment matrix alpha")
        plt.savefig("alignment.png", dpi=120)
        print("saved alignment.png  (compare with Figure 5 in the paper)")
    except ImportError:
        print("matplotlib not installed - skipped the alignment plot")

"""
EXERCISES
  1. Predict: at which T does the no-attention model fall apart? Sweep --T 5 10 20 30 and plot whole-sequence accuracy for both.
     That plot is the paper's 'bottleneck' claim turned into a measurement.
  2. Attention weights are over positions, not content. Replace additive with dot-product (Table 1). Does it still learn the
     anti-diagonal? What does the model use to know 'where' it is?
  3. Sanity check, always: confirm that alphas.sum(1) == 1 (softmax over the right dim!). Then deliberately use dim=1 in
     the softmax. Does anything crash? Does it still train? What do you learn about silent shape bugs?
  4. Task variant: sort the digits instead of reversing. Is the alignment still clean? Why/why not?
  5. Appendix B discards the first RNN output and feeds H_4 to attention. Our code uses S_{t-1} for the score (Bahdanau).
     Change to use the NEW S_t (Luong-style) - what changes in the loop order?
"""
