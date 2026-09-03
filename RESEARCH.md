# Input-adaptive nanoGPT

## Question

Can an autoregressive model receive **less identity-level input** while retaining enough predictive ability? The intervention belongs at ingestion time; it is not post-hoc activation quantization and it is not generating every token then discarding some output.

For each position, the model receives either its normal token embedding or a shared learned `coarsened_input` embedding. It still predicts the next token at every position, so ordinary cross-entropy remains comparable to the dense control. The initial `periodic` policy is content-independent and causal.

## What this first stage does and does not test

| Tests | Does not test yet |
| --- | --- |
| Loss degradation as identity observations are removed before the Transformer | Attention FLOP reduction; sequence length is unchanged |
| Whether a fixed information budget preserves predictive structure | A learned, semantically meaningful reader/controller |
| A matched rate–loss curve | Human eye movements or Japanese reading directly |

A content-aware policy that sees a future token, or a comparison changing target count/model size/training budget, can create a meaningless apparent win.

## Protocol

Prepare the corpus, then run the matched dense and 50% input-budget controls:

```sh
uv run python data/shakespeare_char/prepare.py
uv run python train.py config/train_shakespeare_dense_control.py
uv run python train.py config/train_shakespeare_input_budget.py
uv run pytest
```

Sweep `input_keep_rate` over `1.0, 0.75, 0.5, 0.25`, reporting validation loss against actual retained fraction, wall time, and GPU memory. A single run is not a cognitive claim.

## Next architectural fork

After the rate–loss control, a controller can choose retention from **only earlier retained state**. That tests whether content-sensitive allocation beats the periodic curve at the same rate. Claiming Transformer compute savings needs a second, variable-length packed-sequence architecture that predicts all original targets without exposing tokens from the target span to the packed representation.

## Open decision

The first learned controller needs an explicit unit of analysis: (a) Japanese character/byte streams, (b) an existing BPE stream, or (c) learned byte-to-latent units. These make materially different claims and should not be conflated.
