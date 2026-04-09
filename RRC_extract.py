
import argparse
import csv
import gc
import random
import time

import pandas as pd
import torch
import torch.nn.functional as F
from decimal import Decimal, getcontext, ROUND_FLOOR, ROUND_HALF_DOWN
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Decimal precision
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
# Algorithm 4, Lines 7-9, 10-11
# Interval narrowing for a *known* token
# ---------------------------------------------------------------------------
def interval_narrow(probs: torch.Tensor, L: Decimal, R: Decimal,
                    token_index: int):
    """Narrow [L, R) to the sub-interval corresponding to *token_index*.

    Uses torch.cumsum on **CPU float64** for deterministic boundary
    computation across devices (CUDA cumsum can introduce rounding
    differences that break the encode/decode round-trip).

    Parameters
    ----------
    probs : Tensor
        Top-k probability vector.
    L, R : Decimal
        Current interval bounds.
    token_index : int
        Position of the known token in the top-k list.

    Returns
    -------
    L_new, R_new : Decimal
        Narrowed interval.
    """
    probs_cpu = probs.cpu().to(torch.float64)
    cumulative = torch.cumsum(probs_cpu, dim=0)
    cumulative = cumulative / cumulative[-1]
    boundaries = torch.cat([torch.zeros(1, dtype=torch.float64), cumulative])
    boundaries_np = boundaries.numpy()

    Delta = R - L
    i = token_index
    L_new = L + Decimal(str(boundaries_np[i])) * Delta
    R_new = L + Decimal(str(boundaries_np[i + 1])) * Delta
    return L_new, R_new


# ---------------------------------------------------------------------------
# Rounding helper (Algorithm 4, Line 16)
# ---------------------------------------------------------------------------
def round_half_down(value: Decimal) -> Decimal:
    """Round *value* to an integer, rounding halves toward the smaller value.

    For example, 2.5 → 2, 3.5 → 3.  This matches the ``round_half_down``
    semantics in Algorithm 4, Line 16.
    """
    return value.to_integral_value(rounding=ROUND_HALF_DOWN)


# ---------------------------------------------------------------------------
# Algorithm 4 — full extraction with KV-cache acceleration
# ---------------------------------------------------------------------------
def decode(idx, context_tokens, stegotext_tokens, topk, prng,
           *, model, args):
    """Extract the secret bit-string from *stegotext_tokens*.

    Parameters
    ----------
    idx : int
        Sample index (for logging).
    context_tokens : Tensor
        Tokenised prompt, shape (1, prompt_len).
    stegotext_tokens : Tensor
        Tokenised full stegotext (prompt + generated), shape (1, total_len).
    topk : int
        Top-k truncation size.
    prng : random.Random
        Seeded PRNG instance (same key K used during embedding).
    model : PreTrainedModel
    args : argparse.Namespace

    Returns
    -------
    (extracted_bits, duration) on success, or None on failure.
    """
    start_time = time.time()

    MAX_VAL = Decimal(2 ** args.bit_length)
    L, R = Decimal(0), MAX_VAL                     # Algorithm 4, Line 3

    context_len = context_tokens.shape[1]
    total_len = stegotext_tokens.shape[1]
    t_end = total_len - context_len - 1            # Line 4

    if t_end <= 0:
        print(f"[{idx}] Error: stegotext has no generated tokens beyond context.")
        return None

    # Storage for Phase 2 reverse rotation
    L_history = []       # L^{t-1} at each step
    Delta_history = []   # Delta^{t-1} at each step
    o_history = []       # o^{t} at each step

    # ---- Phase 1: Forward pass with KV-cache -----------------------------
    max_ctx_len = 1000
    input_ids = context_tokens[:, -max_ctx_len:]
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
    past_kv = outputs.past_key_values
    cached_len = input_ids.shape[1]

    # --- t = 0 ------------------------------------------------------------
    logits_sorted, indices = (outputs.logits[0, -1, :]
                              .to(torch.float64).sort(descending=True, stable=True))
    probs = F.softmax(logits_sorted, dim=-1)

    L_history.append(L)
    Delta_history.append(R - L)
    o_history.append(Decimal(prng.random()))

    top_probs = probs[:topk]
    top_tokens = indices[:topk]

    known_id = stegotext_tokens[0, context_len].item()
    pos = (top_tokens == known_id).nonzero(as_tuple=True)[0]
    if len(pos) == 0:
        print(f"[{idx}] Error at t=0: token {known_id} not in top-{topk}")
        return None

    L, R = interval_narrow(top_probs, L, R, pos[0].item())
    next_token = top_tokens[pos[0].item()]
    cached_len += 1

    # --- t = 1, …, t_end -------------------------------------------------
    for t in range(1, t_end + 1):
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
                                  .to(torch.float64).sort(descending=True, stable=True))
        probs = F.softmax(logits_sorted, dim=-1)

        L_history.append(L)
        Delta_history.append(R - L)
        o_history.append(Decimal(prng.random()))

        top_probs = probs[:topk]
        top_tokens = indices[:topk]

        known_id = stegotext_tokens[0, context_len + t].item()
        pos = (top_tokens == known_id).nonzero(as_tuple=True)[0]
        if len(pos) == 0:
            print(f"[{idx}] Error at t={t}: token {known_id} not in top-{topk}")
            return None

        L, R = interval_narrow(top_probs, L, R, pos[0].item())
        next_token = top_tokens[pos[0].item()]

    # ---- Phase 2: Reverse rotation (Lines 12-15) -------------------------
    mid = (L + R) / 2                              # Line 12

    for t in range(t_end, -1, -1):                 # Line 13
        L_prev = L_history[t]
        Delta_prev = Delta_history[t]
        o_t = o_history[t]
        mid = L_prev + decimal_mod(                # Line 15
            mid - L_prev - o_t * Delta_prev, Delta_prev)

    d_s = round_half_down(mid)                     # Line 16
    extracted_bits = bin(int(d_s))[2:].zfill(args.bit_length)  # Line 17

    duration = time.time() - start_time
    return extracted_bits, duration


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="RRC Steganography — Extraction (Algorithm 4)")
    parser.add_argument("--language_model", type=str,
                        default="meta-llama/Llama-3.1-8B",
                        help="HuggingFace model identifier (must match embedding)")
    parser.add_argument("--bit_length", type=int, default=128,
                        help="Length of the secret message in bits")
    parser.add_argument("--top_k", type=int, default=-1,
                        help="Top-k truncation (-1 = full vocabulary)")
    parser.add_argument("--temp", type=float, default=1.0,
                        help="Sampling temperature (reserved)")
    parser.add_argument("--key", type=int, default=42,
                        help="Symmetric key K (PRNG seed)")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to TSV generated by encode_range_kvcache.py")
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

    # ---- Read stegotexts & extract ---------------------------------------
    input_df = pd.read_csv(args.input_file, sep="\t")
    success, fail = 0, 0

    for _, row in input_df.iterrows():
        idx = row["Idx"]
        context_tokens = tokenizer(
            row["Context"], return_tensors="pt"
        ).to(model.device)["input_ids"]
        stego_tokens = tokenizer(
            row["Text"], return_tensors="pt"
        ).to(model.device)["input_ids"]

        prng = random.Random(args.key)

        result = decode(idx, context_tokens, stego_tokens, topk, prng,
                        model=model, args=args)

        if result is not None:
            bits, dur = result
            print(f"[{idx}] Extracted: {bits}  ({dur:.4f}s)")
            success += 1
        else:
            print(f"[{idx}] Extraction failed.")
            fail += 1

        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nDone. Success: {success}, Failed: {fail}")


if __name__ == "__main__":
    main()
