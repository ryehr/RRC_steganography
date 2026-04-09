#!/usr/bin/env python3
"""
End-to-end round-trip test for Rotation Range-Coding (RRC) Steganography.

Embeds a fixed 128-bit secret message via Algorithm 3, extracts it via
Algorithm 4, and verifies that the extracted bits match the original.

This script is self-contained: it bundles trimmed versions of the encode
and decode logic so it can run independently of the main pipeline files.

Usage:
    python test_roundtrip.py

By default the test uses ``openai-community/gpt2`` (locally cached, small).
The algorithm is model-agnostic — correctness depends only on both sides
using the *same* model and key.
"""

import gc
import random
import time

import torch
import torch.nn.functional as F
from decimal import Decimal, getcontext, ROUND_FLOOR, ROUND_HALF_DOWN
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "openai-community/gpt2"   # use a small model for fast CI
BIT_LENGTH = 128
KEY = 42
TOP_K = -1                              # -1 → full vocabulary
CONTEXT_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "In the heart of the ancient forest,"
)

# Fixed secret for reproducibility
SECRET_BITS = (
    "10110011001010111010011100101011"
    "11001011001010111010011100101011"
    "11001011001010111010011100101011"
    "11001011001010111010011100101011"
)
assert len(SECRET_BITS) == BIT_LENGTH

getcontext().prec = 50
MAX_VAL = Decimal(2 ** BIT_LENGTH)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def decimal_mod(a: Decimal, m: Decimal) -> Decimal:
    """True mathematical modulo — result always in [0, m)."""
    return a - m * (a / m).to_integral_value(rounding=ROUND_FLOOR)


def _build_boundaries(probs, L, R):
    """Return (boundaries_np, Delta) for a probability vector."""
    cumulative = torch.cumsum(probs, dim=0)
    cumulative = cumulative / cumulative[-1]
    bnd = torch.cat([torch.zeros(1, device=probs.device), cumulative])
    return bnd.cpu().numpy(), R - L


# ---------------------------------------------------------------------------
# Encode helpers (Algorithm 3)
# ---------------------------------------------------------------------------
def _encode_select(probs, L, R, d_s):
    bnd, Delta = _build_boundaries(probs, L, R)
    for i in range(len(bnd) - 1):
        upper = L + Decimal(str(bnd[i + 1])) * Delta
        if upper > d_s:
            return i, L + Decimal(str(bnd[i])) * Delta, upper
    raise RuntimeError("interval_select failed")


def _check_termination(L, R, d_s):
    diff = (L + R) / 2 - d_s
    return Decimal("-0.5") < diff <= Decimal("0.5")


def encode(context_tokens, topk, prng, model):
    """Run Algorithm 3.  Returns (all_token_ids, generated_count)."""
    d_s = Decimal(int(SECRET_BITS, 2))
    L, R = Decimal(0), MAX_VAL

    max_ctx = 1000
    input_ids = context_tokens[:, -max_ctx:]
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
    past_kv = out.past_key_values
    cached = input_ids.shape[1]

    all_ids = context_tokens.clone()
    n_tok = 0

    # t = 0
    ls, idx = out.logits[0, -1, :].cpu().to(torch.float64).sort(descending=True)
    pr = F.softmax(ls, dim=-1)
    o = Decimal(prng.random())
    Delta = R - L
    d_s = L + decimal_mod((d_s - L) + o * Delta, Delta)
    si, L, R = _encode_select(pr[:topk], L, R, d_s)
    nxt = idx[:topk][si].to(model.device)
    all_ids = torch.cat([all_ids, nxt.unsqueeze(0).unsqueeze(0)], dim=-1)
    n_tok += 1
    cached += 1
    if _check_termination(L, R, d_s):
        return all_ids, n_tok

    while True:
        if n_tok >= 2 * BIT_LENGTH:
            raise RuntimeError("Encoding did not terminate within token limit")
        inp = nxt.unsqueeze(0).unsqueeze(0)
        if cached > max_ctx:
            trim = cached - max_ctx
            past_kv = tuple((k[:, :, trim:, :], v[:, :, trim:, :])
                            for k, v in past_kv)
            cached = max_ctx
        with torch.no_grad():
            out = model(inp, past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values
        cached += 1

        ls, idx = out.logits[0, -1, :].cpu().to(torch.float64).sort(descending=True)
        pr = F.softmax(ls, dim=-1)
        o = Decimal(prng.random())
        Delta = R - L
        d_s = L + decimal_mod((d_s - L) + o * Delta, Delta)
        si, L, R = _encode_select(pr[:topk], L, R, d_s)
        nxt = idx[:topk][si].to(model.device)
        all_ids = torch.cat([all_ids, nxt.unsqueeze(0).unsqueeze(0)], dim=-1)
        n_tok += 1
        if _check_termination(L, R, d_s):
            break

    return all_ids, n_tok


# ---------------------------------------------------------------------------
# Decode helpers (Algorithm 4)
# ---------------------------------------------------------------------------
def _decode_narrow(probs, L, R, token_idx):
    bnd, Delta = _build_boundaries(probs, L, R)
    return (L + Decimal(str(bnd[token_idx])) * Delta,
            L + Decimal(str(bnd[token_idx + 1])) * Delta)


def decode(context_tokens, stego_tokens, topk, prng, model):
    """Run Algorithm 4.  Returns extracted_bits string or None."""
    L, R = Decimal(0), MAX_VAL
    ctx_len = context_tokens.shape[1]
    t_end = stego_tokens.shape[1] - ctx_len - 1
    if t_end <= 0:
        print("Error: no generated tokens beyond context.")
        return None

    L_hist, D_hist, o_hist = [], [], []
    max_ctx = 1000
    input_ids = context_tokens[:, -max_ctx:]
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
    past_kv = out.past_key_values
    cached = input_ids.shape[1]

    # t = 0
    ls, idx = out.logits[0, -1, :].cpu().to(torch.float64).sort(descending=True)
    pr = F.softmax(ls, dim=-1)
    L_hist.append(L); D_hist.append(R - L); o_hist.append(Decimal(prng.random()))
    top_pr, top_tk = pr[:topk], idx[:topk]
    kid = stego_tokens[0, ctx_len].item()
    pos = (top_tk == kid).nonzero(as_tuple=True)[0]
    if len(pos) == 0:
        return None
    L, R = _decode_narrow(top_pr, L, R, pos[0].item())
    nxt = top_tk[pos[0].item()].to(model.device)
    cached += 1

    for t in range(1, t_end + 1):
        inp = nxt.unsqueeze(0).unsqueeze(0)
        if cached > max_ctx:
            trim = cached - max_ctx
            past_kv = tuple((k[:, :, trim:, :], v[:, :, trim:, :])
                            for k, v in past_kv)
            cached = max_ctx
        with torch.no_grad():
            out = model(inp, past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values
        cached += 1

        ls, idx = out.logits[0, -1, :].cpu().to(torch.float64).sort(descending=True)
        pr = F.softmax(ls, dim=-1)
        L_hist.append(L); D_hist.append(R - L); o_hist.append(Decimal(prng.random()))
        top_pr, top_tk = pr[:topk], idx[:topk]
        kid = stego_tokens[0, ctx_len + t].item()
        pos = (top_tk == kid).nonzero(as_tuple=True)[0]
        if len(pos) == 0:
            return None
        L, R = _decode_narrow(top_pr, L, R, pos[0].item())
        nxt = top_tk[pos[0].item()].to(model.device)

    # Phase 2: reverse rotation
    mid = (L + R) / 2
    for t in range(t_end, -1, -1):
        mid = L_hist[t] + decimal_mod(
            mid - L_hist[t] - o_hist[t] * D_hist[t], D_hist[t])

    d_s = mid.to_integral_value(rounding=ROUND_HALF_DOWN)
    return bin(int(d_s))[2:].zfill(BIT_LENGTH)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    sep = "=" * 60
    print(sep)
    print("Round-trip Test: Rotation RC Steganography")
    print(sep)
    print(f"  Model:      {MODEL_NAME}")
    print(f"  Bit length: {BIT_LENGTH}")
    print(f"  Key:        {KEY}")
    print(f"  Secret:     {SECRET_BITS}")
    print()

    # Device selection
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"  Device:     {device}")

    # Load model
    print("  Loading model …")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, TOKENIZERS_PARALLELISM=True, padding_side="left")
    model.resize_token_embeddings(len(tokenizer.vocab))
    model.eval()

    vocab_size = len(tokenizer.vocab)
    topk = vocab_size if TOP_K < 0 else TOP_K
    print(f"  Vocab size: {vocab_size}, top_k: {topk}")

    context_tokens = tokenizer(
        CONTEXT_TEXT, return_tensors="pt"
    ).to(model.device)["input_ids"]
    print(f"  Context tokens: {context_tokens.shape[1]}")
    print()

    # ---- Step 1: Encode --------------------------------------------------
    print(">>> Step 1: Encoding …")
    t0 = time.time()
    all_ids, gen_count = encode(context_tokens, topk,
                                random.Random(KEY), model)
    t_enc = time.time() - t0
    stegotext = tokenizer.decode(all_ids.squeeze(0), skip_special_tokens=True)
    print(f"    Generated tokens: {gen_count}")
    print(f"    Encode time:      {t_enc:.2f} s")
    print(f"    Stegotext:        {stegotext[:200]} …")
    print()

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # ---- Step 2: Decode --------------------------------------------------
    print(">>> Step 2: Decoding (extracting) …")
    stego_tokens = tokenizer(
        stegotext, return_tensors="pt"
    ).to(model.device)["input_ids"]
    ctx_tokens_dec = tokenizer(
        CONTEXT_TEXT, return_tensors="pt"
    ).to(model.device)["input_ids"]

    t0 = time.time()
    extracted = decode(ctx_tokens_dec, stego_tokens, topk,
                       random.Random(KEY), model)
    t_dec = time.time() - t0
    print(f"    Decode time:      {t_dec:.2f} s")
    print()

    # ---- Step 3: Verify --------------------------------------------------
    print(">>> Step 3: Verification")
    print(f"    Original: {SECRET_BITS}")
    print(f"    Extracted: {extracted}")

    if extracted == SECRET_BITS:
        print("\n    ✅ SUCCESS — extracted message matches perfectly!")
    else:
        print("\n    ❌ FAILURE — mismatch!")
        if extracted is not None:
            for i, (a, b) in enumerate(zip(SECRET_BITS, extracted)):
                if a != b:
                    print(f"    First diff at bit {i}: expected '{a}', got '{b}'")
                    break

    print()
    print(sep)
    print("Test complete.")


if __name__ == "__main__":
    main()
