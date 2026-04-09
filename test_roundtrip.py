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


# ---------------------------------------------------------------------------
# Encode helpers (Algorithm 3)
# ---------------------------------------------------------------------------
def _encode_select(probs, L, R, d_s):
    """Narrow [L, R) to the sub-interval containing d_s; return (index, L_new, R_new).

    Uses torch.cumsum on CPU float64 for deterministic boundaries, and only
    converts to Decimal at the hit point.
    """
    # Force cumsum on CPU float64 for deterministic results across devices
    probs_cpu = probs.cpu().to(torch.float64)
    cumulative = torch.cumsum(probs_cpu, dim=0)
    cumulative = cumulative / cumulative[-1]
    bnd = torch.cat([torch.zeros(1, dtype=torch.float64), cumulative])
    bnd_np = bnd.numpy()
    Delta = R - L

    for i in range(len(bnd_np) - 1):
        if L + Decimal(str(bnd_np[i + 1])) * Delta > d_s:
            return i, L + Decimal(str(bnd_np[i])) * Delta, \
                      L + Decimal(str(bnd_np[i + 1])) * Delta
    raise RuntimeError("interval_select failed")


def _check_termination(L, R, d_s):
    diff = (L + R) / 2 - d_s
    return Decimal("-0.5") < diff <= Decimal("0.5")


def encode(context_tokens, topk, prng, model, debug=False):
    """Run Algorithm 3.  Returns (all_token_ids, generated_count, debug_log)."""
    d_s = Decimal(int(SECRET_BITS, 2))
    L, R = Decimal(0), MAX_VAL
    log = []  # per-step debug snapshots

    max_ctx = 1000
    input_ids = context_tokens[:, -max_ctx:]
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
    past_kv = out.past_key_values
    cached = input_ids.shape[1]

    generated_ids = []  # collect token ids in a list (fast append)

    # t = 0
    logits_sorted, idx = (out.logits[0, -1, :]
                          .to(torch.float64).sort(descending=True, stable=True))
    pr = F.softmax(logits_sorted, dim=-1)
    o = Decimal(prng.random())
    Delta = R - L
    d_s = L + decimal_mod((d_s - L) + o * Delta, Delta)
    si, L, R = _encode_select(pr[:topk], L, R, d_s)
    nxt = idx[:topk][si]
    generated_ids.append(nxt.item())
    if debug:
        log.append(dict(t=0, token=nxt.item(), L=L, R=R, o=o,
                        logits_hash=logits_sorted[:5].tolist()))
    cached += 1
    if _check_termination(L, R, d_s):
        all_ids = torch.cat([context_tokens,
                             torch.tensor([generated_ids], device=model.device)],
                            dim=-1)
        return all_ids, len(generated_ids), log

    step = 1
    while True:
        if len(generated_ids) >= 2 * BIT_LENGTH:
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

        logits_sorted, idx = (out.logits[0, -1, :]
                              .to(torch.float64).sort(descending=True, stable=True))
        pr = F.softmax(logits_sorted, dim=-1)
        o = Decimal(prng.random())
        Delta = R - L
        d_s = L + decimal_mod((d_s - L) + o * Delta, Delta)
        si, L, R = _encode_select(pr[:topk], L, R, d_s)
        nxt = idx[:topk][si].to(model.device)
        generated_ids.append(nxt.item())
        if debug:
            log.append(dict(t=step, token=nxt.item(), L=L, R=R, o=o,
                            logits_hash=logits_sorted[:5].tolist()))
        step += 1
        if _check_termination(L, R, d_s):
            break

    all_ids = torch.cat([context_tokens,
                         torch.tensor([generated_ids], device=model.device)],
                        dim=-1)
    return all_ids, len(generated_ids), log


# ---------------------------------------------------------------------------
# Decode helpers (Algorithm 4)
# ---------------------------------------------------------------------------
def _decode_narrow(probs, L, R, token_idx):
    """Narrow [L, R) to the sub-interval for token_idx.

    Uses torch.cumsum on CPU float64 for deterministic boundaries.
    """
    probs_cpu = probs.cpu().to(torch.float64)
    cumulative = torch.cumsum(probs_cpu, dim=0)
    cumulative = cumulative / cumulative[-1]
    bnd = torch.cat([torch.zeros(1, dtype=torch.float64), cumulative])
    bnd_np = bnd.numpy()
    Delta = R - L
    return (L + Decimal(str(bnd_np[token_idx])) * Delta,
            L + Decimal(str(bnd_np[token_idx + 1])) * Delta)


def decode(context_tokens, stego_tokens, topk, prng, model, debug=False):
    """Run Algorithm 4.  Returns extracted_bits string or None."""
    L, R = Decimal(0), MAX_VAL
    ctx_len = context_tokens.shape[1]
    t_end = stego_tokens.shape[1] - ctx_len - 1
    if t_end <= 0:
        print("Error: no generated tokens beyond context.")
        return None, []

    log = []
    L_hist, D_hist, o_hist = [], [], []
    max_ctx = 1000
    input_ids = context_tokens[:, -max_ctx:]
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
    past_kv = out.past_key_values
    cached = input_ids.shape[1]

    # t = 0
    logits_sorted, idx = (out.logits[0, -1, :]
                          .to(torch.float64).sort(descending=True, stable=True))
    pr = F.softmax(logits_sorted, dim=-1)
    o_val = Decimal(prng.random())
    L_hist.append(L); D_hist.append(R - L); o_hist.append(o_val)
    top_pr, top_tk = pr[:topk], idx[:topk]
    kid = stego_tokens[0, ctx_len].item()
    pos = (top_tk == kid).nonzero(as_tuple=True)[0]
    if len(pos) == 0:
        print(f"    [DEBUG] t=0: token {kid} not in top-{topk}!")
        return None, []
    L, R = _decode_narrow(top_pr, L, R, pos[0].item())
    if debug:
        log.append(dict(t=0, token=kid, L=L, R=R, o=o_val,
                        logits_hash=logits_sorted[:5].tolist()))
    nxt = top_tk[pos[0].item()]
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

        logits_sorted, idx = (out.logits[0, -1, :]
                              .to(torch.float64).sort(descending=True, stable=True))
        pr = F.softmax(logits_sorted, dim=-1)
        o_val = Decimal(prng.random())
        L_hist.append(L); D_hist.append(R - L); o_hist.append(o_val)
        top_pr, top_tk = pr[:topk], idx[:topk]
        kid = stego_tokens[0, ctx_len + t].item()
        pos = (top_tk == kid).nonzero(as_tuple=True)[0]
        if len(pos) == 0:
            print(f"    [DEBUG] t={t}: token {kid} not in top-{topk}!")
            return None, []
        L, R = _decode_narrow(top_pr, L, R, pos[0].item())
        if debug:
            log.append(dict(t=t, token=kid, L=L, R=R, o=o_val,
                            logits_hash=logits_sorted[:5].tolist()))
        nxt = top_tk[pos[0].item()]

    # Phase 2: reverse rotation
    mid = (L + R) / 2
    for t in range(t_end, -1, -1):
        mid = L_hist[t] + decimal_mod(
            mid - L_hist[t] - o_hist[t] * D_hist[t], D_hist[t])

    d_s = mid.to_integral_value(rounding=ROUND_HALF_DOWN)
    return bin(int(d_s))[2:].zfill(BIT_LENGTH), log


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
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    all_ids, gen_count, enc_log = encode(context_tokens, topk,
                                         random.Random(KEY), model,
                                         debug=True)
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
    t0 = time.time()
    extracted, dec_log = decode(context_tokens, all_ids, topk,
                                random.Random(KEY), model, debug=True)
    t_dec = time.time() - t0
    print(f"    Decode time:      {t_dec:.2f} s")
    print()

    # ---- Step 3: Verify --------------------------------------------------
    print(">>> Step 3: Verification")
    print(f"    Original:  {SECRET_BITS}")
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

        # ---- Debug: per-step comparison ----------------------------------
        print("\n>>> Debug: per-step encode vs decode comparison")
        n_steps = min(len(enc_log), len(dec_log))
        for s in range(n_steps):
            e, d = enc_log[s], dec_log[s]
            tok_match = e["token"] == d["token"]
            L_match = e["L"] == d["L"]
            R_match = e["R"] == d["R"]
            o_match = e["o"] == d["o"]
            logits_match = e["logits_hash"] == d["logits_hash"]
            if not (tok_match and L_match and R_match and o_match and logits_match):
                print(f"    *** FIRST DIVERGENCE at t={e['t']} ***")
                print(f"    token:  enc={e['token']}  dec={d['token']}  {'✓' if tok_match else '✗'}")
                print(f"    L:      {'✓' if L_match else '✗'}")
                if not L_match:
                    print(f"      enc: {e['L']}")
                    print(f"      dec: {d['L']}")
                print(f"    R:      {'✓' if R_match else '✗'}")
                if not R_match:
                    print(f"      enc: {e['R']}")
                    print(f"      dec: {d['R']}")
                print(f"    o:      {'✓' if o_match else '✗'}")
                if not o_match:
                    print(f"      enc: {e['o']}")
                    print(f"      dec: {d['o']}")
                print(f"    logits: {'✓' if logits_match else '✗'}")
                if not logits_match:
                    print(f"      enc: {e['logits_hash']}")
                    print(f"      dec: {d['logits_hash']}")
                break
        else:
            if len(enc_log) != len(dec_log):
                print(f"    All {n_steps} common steps match, but step counts differ: "
                      f"enc={len(enc_log)}, dec={len(dec_log)}")
            else:
                print(f"    All {n_steps} steps match — divergence may be in Phase 2 (reverse rotation)")

    print()
    print(sep)
    print(f"Test complete.  Total: {t_enc + t_dec:.2f} s "
          f"(encode {t_enc:.2f} s + decode {t_dec:.2f} s)")


if __name__ == "__main__":
    main()
