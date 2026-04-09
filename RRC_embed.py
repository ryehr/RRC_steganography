#!/usr/bin/env python3
"""
Rotation Range-Coding (RRC) Steganography — Embedding (Algorithm 3)

This module implements the embedding procedure described in Algorithm 3 of:
  "Efficient Provably Secure Linguistic Steganography via Range Coding"
  (Yan & Murawaki, 2025)

Given a secret bit-string m_s, a language model M, and a shared symmetric key K,
the encoder produces a steganographic text t_s that is statistically
indistinguishable from normal LM output while embedding m_s at near-100%
entropy utilization.

Usage:
    python encode_range_kvcache.py \\
        --language_model meta-llama/Llama-3.1-8B \\
        --bit_length 128 \\
        --key 42 \\
        [--top_k -1] [--part 0] [--part_max 2]

Requires:
    - A prompts file  ``0.Prompts.tsv``  with columns ``idx`` and ``text``.
    - Output is written to a TSV whose name encodes the model and bit length.
"""

import argparse
import csv
import gc
import os
import random
import time

import pandas as pd
import torch
import torch.nn.functional as F
from decimal import Decimal, getcontext, ROUND_FLOOR
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Decimal precision — must be high enough to represent 2^bit_length exactly
# ---------------------------------------------------------------------------
getcontext().prec = 50


# ---------------------------------------------------------------------------
# Utility: correct modulo for Python Decimal
# ---------------------------------------------------------------------------
def decimal_mod(a: Decimal, m: Decimal) -> Decimal:
    """Return *a* mod *m* with the result always in [0, m).

    Python's built-in ``Decimal.__mod__`` uses *truncated* division, which can
    return negative remainders for negative dividends.  This helper uses
    floor-division semantics instead.
    """
    return a - m * (a / m).to_integral_value(rounding=ROUND_FLOOR)


# ---------------------------------------------------------------------------
# Algorithm 3, Lines 6-8, 11-12
# Interval construction, token selection, and interval update
# ---------------------------------------------------------------------------
def interval_select(probs: torch.Tensor, L: Decimal, R: Decimal,
                    d_s: Decimal):
    """Narrow [L, R) to the sub-interval that contains *d_s*.

    Parameters
    ----------
    probs : Tensor
        Top-k probability vector (already sorted descending).
    L, R : Decimal
        Current interval bounds.
    d_s : Decimal
        Rotated secret decimal value.

    Returns
    -------
    selected_index : int
        Index of the chosen token in the top-k list.
    L_new, R_new : Decimal
        Updated interval bounds after selection.
    """
    cumulative = torch.cumsum(probs, dim=0)
    cumulative = cumulative / cumulative[-1]          # normalise
    boundaries = torch.cat([torch.zeros(1, device=probs.device), cumulative])
    boundaries_np = boundaries.cpu().numpy()

    Delta = R - L

    for i in range(len(boundaries_np) - 1):
        upper = L + Decimal(str(boundaries_np[i + 1])) * Delta
        if upper > d_s:
            L_new = L + Decimal(str(boundaries_np[i])) * Delta
            R_new = upper
            return i, L_new, R_new

    raise RuntimeError("d_s not covered by any sub-interval (should not happen)")


# ---------------------------------------------------------------------------
# Algorithm 3, Lines 14-15 — termination check
# ---------------------------------------------------------------------------
def check_termination(L: Decimal, R: Decimal, d_s: Decimal) -> bool:
    """Return True when (L + R) / 2 − d_s ∈ (−0.5, 0.5]."""
    diff = (L + R) / 2 - d_s
    return Decimal("-0.5") < diff <= Decimal("0.5")


# ---------------------------------------------------------------------------
# Algorithm 3 — full embedding with KV-cache acceleration
# ---------------------------------------------------------------------------
def encode(idx, context_text, context_tokens, topk, prng,
           *, model, tokenizer, args, secret_bits, output_path):
    """Embed *secret_bits* into LM-generated text and write one TSV row.

    Parameters
    ----------
    idx : int
        Sample index (written to the output file).
    context_text : str
        Raw prompt string.
    context_tokens : Tensor
        Tokenised prompt, shape (1, seq_len).
    topk : int
        Top-k truncation size (use full vocab when set to ``len(vocab)``).
    prng : random.Random
        Seeded PRNG instance for rotation offsets.
    model : PreTrainedModel
    tokenizer : PreTrainedTokenizer
    args : argparse.Namespace
    secret_bits : str
        The l-bit secret message as a ``'0'``/``'1'`` string.
    output_path : str
        Path to the output TSV file.

    Returns
    -------
    bool — True on successful embedding, False if the token budget is exceeded.
    """
    start_time = time.time()

    MAX_VAL = Decimal(2 ** args.bit_length)
    d_s = Decimal(int(secret_bits, 2))             # Algorithm 3, Line 1

    L, R = Decimal(0), MAX_VAL                     # Algorithm 3, Line 3
    token_total = 0
    entropy_total = 0

    # ---- KV-cache prefill ------------------------------------------------
    max_ctx_len = 1000
    input_ids = context_tokens[:, -max_ctx_len:]
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
    past_kv = outputs.past_key_values
    cached_len = input_ids.shape[1]

    # ---- t = 0  (first token from prefill logits) ------------------------
    logits_sorted, indices = (outputs.logits[0, -1, :]
                              .to(torch.float64).sort(descending=True))
    probs = F.softmax(logits_sorted, dim=-1)
    entropy_total += -torch.sum(probs * torch.log2(probs + 1e-10))

    o = Decimal(prng.random())                     # Line 9
    Delta = R - L
    d_s = L + decimal_mod((d_s - L) + o * Delta, Delta)   # Line 10
    assert L <= d_s < R

    top_probs = probs[:topk]
    top_tokens = indices[:topk]
    sel_idx, L, R = interval_select(top_probs, L, R, d_s)  # Lines 6-8, 11-12

    next_token = top_tokens[sel_idx]
    all_ids = torch.cat([context_tokens,
                         next_token.unsqueeze(0).unsqueeze(0)], dim=-1)
    token_total += 1
    cached_len += 1

    if check_termination(L, R, d_s):               # Lines 14-15
        return _write_result(idx, context_text, all_ids, token_total,
                             entropy_total, start_time, tokenizer, args,
                             output_path)

    # ---- t = 1, 2, … (incremental KV-cache steps) -----------------------
    while True:
        if token_total >= 2 * args.bit_length:
            return False

        new_input = next_token.unsqueeze(0).unsqueeze(0)

        if cached_len > max_ctx_len:
            trim = cached_len - max_ctx_len
            past_kv = tuple(
                (k[:, :, trim:, :], v[:, :, trim:, :]) for k, v in past_kv
            )
            cached_len = max_ctx_len

        with torch.no_grad():
            outputs = model(new_input, past_key_values=past_kv, use_cache=True)
        past_kv = outputs.past_key_values
        cached_len += 1

        logits_sorted, indices = (outputs.logits[0, -1, :]
                                  .to(torch.float64).sort(descending=True))
        probs = F.softmax(logits_sorted, dim=-1)
        entropy_total += -torch.sum(probs * torch.log2(probs + 1e-10))

        o = Decimal(prng.random())                 # Line 9
        Delta = R - L
        d_s = L + decimal_mod((d_s - L) + o * Delta, Delta)  # Line 10
        assert L <= d_s < R

        top_probs = probs[:topk]
        top_tokens = indices[:topk]
        sel_idx, L, R = interval_select(top_probs, L, R, d_s)

        next_token = top_tokens[sel_idx]
        all_ids = torch.cat([all_ids,
                             next_token.unsqueeze(0).unsqueeze(0)], dim=-1)
        token_total += 1

        if check_termination(L, R, d_s):
            break

    return _write_result(idx, context_text, all_ids, token_total,
                         entropy_total, start_time, tokenizer, args,
                         output_path)


def _write_result(idx, context_text, all_ids, token_total, entropy_total,
                  start_time, tokenizer, args, output_path):
    """Compute metrics and append one row to the output TSV."""
    duration = time.time() - start_time
    bpt = args.bit_length / token_total
    avg_entropy = entropy_total / token_total
    utilization = bpt / avg_entropy
    stegotext = tokenizer.decode(all_ids.squeeze(0), skip_special_tokens=True)

    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([idx, 0, bpt, avg_entropy.item(),
                         utilization.item(), duration, "-",
                         context_text, stegotext])
    return True


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="RRC Steganography — Embedding (Algorithm 3)")
    parser.add_argument("--language_model", type=str,
                        default="meta-llama/Llama-3.1-8B",
                        help="HuggingFace model identifier")
    parser.add_argument("--bit_length", type=int, default=128,
                        help="Length of the secret message in bits")
    parser.add_argument("--top_k", type=int, default=-1,
                        help="Top-k truncation (-1 = full vocabulary)")
    parser.add_argument("--temp", type=float, default=1.0,
                        help="Sampling temperature (reserved, not yet used)")
    parser.add_argument("--part", type=int, default=0,
                        help="Partition index for parallel execution")
    parser.add_argument("--part_max", type=int, default=2,
                        help="Total number of partitions")
    parser.add_argument("--key", type=int, default=42,
                        help="Symmetric key K (PRNG seed shared by sender & receiver)")
    args = parser.parse_args()
    print(args)

    # ---- Load model & tokenizer ------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        args.language_model, torch_dtype="auto", device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.language_model, TOKENIZERS_PARALLELISM=True, padding_side="left")
    model.resize_token_embeddings(len(tokenizer.vocab))
    model.eval()

    vocab_size = len(tokenizer.vocab)
    topk = vocab_size if args.top_k < 0 else args.top_k

    # ---- Prepare I/O -----------------------------------------------------
    prompt_df = pd.read_csv("0.Prompts.tsv", sep="\t")
    header = ["Idx", "KL", "BPT", "Entropy", "Utilization",
              "Time", "PPL", "Context", "Text"]
    model_short = args.language_model.rsplit("/", 1)[-1]
    output_path = f"1.RC_decimal_{model_short}_bit{args.bit_length}.tsv"

    if not os.path.exists(output_path):
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f, delimiter="\t").writerow(header)

    # ---- Encode loop -----------------------------------------------------
    for _, row in prompt_df.iterrows():
        idx = row["idx"]
        lo = args.part * len(prompt_df) / args.part_max
        hi = (args.part + 1) * len(prompt_df) / args.part_max
        if idx < lo or idx >= hi:
            continue

        while True:
            secret_bits = "".join(
                str(random.randint(0, 1)) for _ in range(args.bit_length))
            context_tokens = tokenizer(
                row["text"], return_tensors="pt"
            ).to(model.device)["input_ids"]

            prng = random.Random(args.key)

            if encode(idx, row["text"], context_tokens, topk, prng,
                      model=model, tokenizer=tokenizer, args=args,
                      secret_bits=secret_bits, output_path=output_path):
                break

            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
