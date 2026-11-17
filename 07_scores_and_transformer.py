"""
STAGE 7  -  paper section 8.2-8.3 + Appendix C: score functions, scaling, positional encoding, self-attention

From here on the layout is BATCH-FIRST (B, T, D), the convention of the Transformer world.

Parts:
  A. All six score functions of Table 1 as tensor code (one line each) + shape/normalisation tests.
  B. An EXPERIMENT on the scaled dot-product claim.
  C. Sinusoidal positional encodings and a test of the claims made about them.
  D. Masked multi-head self-attention in ~25 lines, with three property tests.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import seed_all, header, check_shape, check_close


# ---------------------------------- A. score functions (Table 1) ----------------------------------
# S: (n, ds) decoder state     H: (T, n, dh) encoder states     ->  scores (T, n)
def score_dot(S, H):
    return (H * S).sum(-1)                                          # (T,n,d)*(n,d) -> (T,n)

def score_scaled_dot(S, H):
    return score_dot(S, H) / S.shape[-1] ** 0.5                     # divide by sqrt(d), the KEY DIMENSION

def score_general(S, H, Wa):                                        # S^T W_a H     Wa: (ds, dh)
    return ((S @ Wa).unsqueeze(0) * H).sum(-1)

def score_cosine(S, H):
    return F.cosine_similarity(H, S.unsqueeze(0).expand_as(H), dim=-1)

def score_additive(S, H, Ws, Wh, v):                                # v^T tanh(Ws S + Wh H)
    return torch.tanh(S @ Ws + H @ Wh) @ v                          # ((n,a) + (T,n,a)) @ (a,) -> (T,n)

def score_location(S, Wa):                                          # softmax(W_a S): never looks at H!   Wa: (ds, T)
    return (S @ Wa).T                                               # (n,T).T -> (T,n)


# ---------------------------------- B. the scaling experiment ----------------------------------
def scaling_experiment(T=10, trials=2000):
    print(f"{'d':>6} {'std(q.k)':>9} | mean MAX attention weight, softmax(score / c) over {T} keys")
    print(f"{'':>6} {'':>9} | {'c = 1':>9} {'c = sqrt(d)':>12} {'c = T':>9}")
    for d in (4, 16, 64, 256, 1024):
        q, k = torch.randn(trials, 1, d), torch.randn(trials, T, d)
        s = (q * k).sum(-1)                                         # (trials, T) raw dot products
        row = [F.softmax(s / c, -1).max(-1).values.mean().item() for c in (1.0, d**0.5, float(T))]
        print(f"{d:>6} {s.std().item():>9.2f} | {row[0]:>9.3f} {row[1]:>12.3f} {row[2]:>9.3f}")


# ---------------------------------- C. positional encoding ----------------------------------
def sinusoidal_pe(T, d):
    pos = torch.arange(T).unsqueeze(1).float()                      # (T, 1)
    i = torch.arange(0, d, 2).float().unsqueeze(0)                  # (1, d/2)
    ang = pos / 10000 ** (i / d)                                    # (T, d/2)   one frequency per pair of dims
    pe = torch.zeros(T, d)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(ang), torch.cos(ang)
    return pe


# ---------------------------------- D. multi-head self-attention ----------------------------------
class SelfAttention(nn.Module):
    def __init__(self, d_model, n_head, causal):
        super().__init__()
        assert d_model % n_head == 0
        self.H, self.dk, self.causal = n_head, d_model // n_head, causal
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, use_sdpa=False):                           # x: (B, T, D)
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)                      # 3 x (B,T,D)
        q, k, v = (z.reshape(B, T, self.H, self.dk).transpose(1, 2) for z in (q, k, v))   # (B,H,T,dk)
        if use_sdpa:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        else:
            att = q @ k.transpose(-2, -1) / self.dk**0.5            # (B,H,T,T)   score[i,j]: how much i reads from j
            if self.causal:
                keep = torch.tril(torch.ones(T, T, dtype=torch.bool))
                att = att.masked_fill(~keep, float("-inf"))         # position i may not look at j > i
            att = F.softmax(att, dim=-1)                            # each row sums to 1
            y = att @ v                                             # (B,H,T,dk)
        y = y.transpose(1, 2).reshape(B, T, D)                      # re-join heads
        return self.proj(y)


if __name__ == "__main__":
    seed_all(0)

    header("A. score functions: shape (T, n), and softmax over T sums to 1")
    T, n, d = 7, 3, 16
    S, H = torch.randn(n, d), torch.randn(T, n, d)
    Wa, Ws, Wh, v, Wloc = torch.randn(d, d), torch.randn(d, 12), torch.randn(d, 12), torch.randn(12), torch.randn(d, T)
    scores = {
        "content (cosine)": score_cosine(S, H), "additive": score_additive(S, H, Ws, Wh, v),
        "location": score_location(S, Wloc), "general": score_general(S, H, Wa),
        "dot": score_dot(S, H), "scaled dot": score_scaled_dot(S, H),
    }
    for name, sc in scores.items():
        check_shape(sc, (T, n), name)
        check_close(F.softmax(sc, dim=0).sum(0), torch.ones(n), f"  softmax sums to 1 ({name})")
    print("[note] score_location takes no H argument at all: it cannot see the content it is aligning. Why would anyone use it?")

    header("B. the paper says the dot product is divided by 'the number of characters / n_source'. Test it.")
    scaling_experiment()
    print("Which column stays flat as d grows? Which one saturates toward 1.0 (one-hot attention, ~zero gradient)?")

    header("C. sinusoidal positional encoding")
    pe = sinusoidal_pe(50, 16)
    check_shape(pe, (50, 16), "PE (T, d)")
    check_close(pe[0], torch.tensor([0.0, 1.0] * 8), "PE[0] = [0,1,0,1,...]")
    sim = pe @ pe.T                                                 # (50, 50)
    check_close(sim[3, 5], sim[10, 12], "PE[p].PE[q] depends only on p-q", atol=1e-4)
    print("similarity of position 0 to positions 1,2,5,10,20,40:", [round(sim[0, j].item(), 2) for j in (1, 2, 5, 10, 20, 40)])
    print("Appendix C says 'distant words have more different encodings'. Is that monotone in this output?")

    header("D. self-attention property tests")
    B, T, D = 2, 9, 32
    x = torch.randn(B, T, D)
    sa = SelfAttention(D, 4, causal=True)
    with torch.no_grad():
        out = sa(x)
        check_shape(out, (B, T, D), "output")
        try:
            check_close(out, sa(x, use_sdpa=True), "manual attention vs F.scaled_dot_product_attention")
        except AttributeError:
            print("[skip] F.scaled_dot_product_attention needs torch >= 2.0")

        x2 = x.clone(); x2[:, 5:] += torch.randn(B, T - 5, D)       # change only the FUTURE (positions >= 5)
        check_close(sa(x2)[:, :5], out[:, :5], "causal: changing t>=5 leaves outputs t<5 untouched")

        sa_full = SelfAttention(D, 4, causal=False)
        perm = torch.randperm(T)
        check_close(sa_full(x[:, perm]), sa_full(x)[:, perm], "NO mask, NO positions: attention is permutation-equivariant")

        pe = sinusoidal_pe(T, D)
        lhs, rhs = sa_full(x[:, perm] + pe), sa_full(x + pe)[:, perm]
        print(f"[info] with positional encoding added: max|diff| = {(lhs - rhs).abs().max().item():.3f}  (should be clearly > 0)")

"""
EXERCISES
  1. Section B: write the prediction for each column BEFORE running. Then explain in one sentence, using Var(q.k) = d for
     unit-variance entries, why sqrt(d) is the right constant. (This is your first 'derive it, then measure it' loop.)
  2. Extend B: also report the mean gradient norm of the softmax output w.r.t. its input for the c=1 column. Does it vanish?
  3. Section C: why is the offset-only property of PE[p].PE[q] true? (Hint: sin a sin b + cos a cos b.) Plot sim[0, :] for d=16 and d=128.
  4. Make the test of D 'break': remove the masked_fill line. Which test fails? Now put the mask AFTER the softmax. Which fails and why?
  5. Implement a 2-layer causal transformer block (attention + MLP + residual + LayerNorm) and train it on stage 3's corpus.
     That is exactly Andrej's makemore->GPT step; you will recognise every tensor shape.
"""
