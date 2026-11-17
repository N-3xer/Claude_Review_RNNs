# From the RNN paper to code

A walk through *"Recurrent Neural Networks (RNNs): A gentle Introduction and Overview"* (Schmidt, 2019), one paper section
per file. Every equation becomes tensor code with the shape in a comment, and every file ends with checks that
compare your understanding against autograd, `torch.nn`, or a number you can compute by hand.

```
pip install -r requirements.txt        # torch >= 2.0 ; matplotlib optional
python 01_rnn_cell.py
```

| file | paper | what you build | how you know it is right |
|---|---|---|---|
| `01_rnn_cell.py` | §1, Eq. 1-4 | the recurrent cell, loop vs hoisted | equals `nn.RNN`; `W_hh=0` gives a feedforward net |
| `02_bptt.py` | §2-3, Eq. 5-12 | BPTT **by hand**, truncated BPTT, grad clipping, vanishing/exploding table | equals autograd and a numerical gradient; table equals `sqrt(h)*rho^lag` |
| `03_char_rnn.py` | §1-3 in practice | makemore-style char LM | initial loss = ln V; overfit one batch; loss vs a computed irreducible floor |
| `04_lstm.py` | §4, Eq. 13-18 | LSTM (paper form and fused form), long-memory experiment | equals `nn.LSTM`; RNN-vs-LSTM accuracy table |
| `05_deep_bidirectional.py` | §5-6, Eq. 19-24 | stacked + bidirectional RNN | equals `nn.RNN(num_layers=2, bidirectional=True)`; look-ahead task |
| `06_seq2seq_attention.py` | §7, §8.1, App. B | encoder-decoder, additive attention | alignment matrix is the anti-diagonal; bottleneck measured |
| `07_scores_and_transformer.py` | §8.2-8.3, App. C | Table 1 score functions, sinusoidal PE, masked multi-head attention | scaling experiment; causal, permutation and PE property tests |
| `08_pointer_net.py` | §9, Eq. 29 | pointer network that sorts | whole-sequence accuracy vs chance 1/T! |

## How to work through a stage (this is the part that builds the skill)
1. **Read the docstring and the equations only.** Close the code.
2. **Write your own version**, shape comment on every line, in a scratch file. Get stuck. Note exactly where in `NOTES.md`.
3. **Run the checks** against your version (swap the class in), then compare with the reference and diff *ideas*, not characters.
4. **Predict before every run** and write the prediction down. A wrong prediction is the gap you are looking for.
5. **Do the EXERCISES at the bottom of the file.** Each is a small hypothesis -> experiment -> result loop.

## Claims in the paper to test (the code is built so you can check each one)
- **Eq. 11-12 (§2):** compare with the recursion in `02_bptt.py`. What multiplies the tanh derivative? Which `H` does the `W_hh` gradient use?
- **Eq. 24 (§6):** "concatenation on axis 0". Is that consistent with `W_ho` being `2h x o`?
- **§3:** "eigenvalues > 1 diverge." Does that hold for tanh as well as for linear? (`02`, section 3)
- **§4:** the output-gate matrix and the RNN readout matrix are both called `W_ho`. Are they the same shape?
- **§8.2:** what is the dot product actually divided by in the Transformer, and why? (`07`, section B)
- **Appendix C:** "binary encoding" and "close words have closer encodings". How true is each? (`07`, section C)

## If something fails
Read the assertion message first: the `check_*` helpers print which tensor, which shape, and how large the difference is.
Shape errors are the most likely cause, and finding them is the exercise. Keep the traceback in `NOTES.md`.
# Claude_Review_RNNs
