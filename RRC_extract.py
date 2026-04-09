import torch
import torch.nn.functional as F
import argparse
import numpy as np
import sys
import math
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import random
import time
import os
import csv
from decimal import *
import pandas as pd
import gc

getcontext().prec = 50


# ============================================================
# Algorithm 4, Lines 7-9, 10-11: Interval construction & token matching & interval update
# (Shared logic with Algorithm 3 encoding, but here the token is known)
# ============================================================
def range_decoding_detail(valid_probabilities, current_min, current_max, known_token_index):
    """
    Corresponds to Algorithm 4:
      Line 7: c^{(t)} <- 0 || p^{(t)}.cumsum()
      Line 8: Delta^{(t-1)} = R^{(t-1)} - L^{(t-1)}
      Line 9: c'^{(t)} <- L^{(t-1)} + Delta^{(t-1)} * c^{(t)}
      Line 10: Select token_i so that token_i = S[|C|+t+1]  (known)
      Line 11: [L^{(t)}, R^{(t)}) <- [c'^{(t)}[i-1], c'^{(t)}[i])

    Args:
        valid_probabilities: top-k probability distribution
        current_min: L^{(t-1)}
        current_max: R^{(t-1)}
        known_token_index: index of the known token within the top-k list

    Returns:
        current_min_new: L^{(t)}
        current_max_new: R^{(t)}
    """
    # Line 7: c^{(t)} <- 0 || p^{(t)}.cumsum()
    cumulative = torch.cumsum(valid_probabilities, dim=0)
    cumulative = cumulative / cumulative[-1]
    assert cumulative[-1].item() == 1.0
    boundaries = torch.cat([torch.tensor([0.0], device=valid_probabilities.device), cumulative[:]])
    boundaries_np = boundaries.cpu().numpy()

    # Line 8: Delta^{(t-1)} = R^{(t-1)} - L^{(t-1)}
    Delta = current_max - current_min

    # Line 9 & 11: Rescale and update interval
    i = known_token_index
    current_min_new = current_min + Decimal(str(boundaries_np[i])) * Delta
    current_max_new = current_min + Decimal(str(boundaries_np[i + 1])) * Delta

    return current_min_new, current_max_new


# ============================================================
# Algorithm 4: Rotation RC steganography (extract)
# ============================================================
def decode_range(idx, context_text, context, stegotext_tokens, topk, prng):
    """
    Full extraction procedure corresponding to Algorithm 4, with KV cache optimization.

    The extraction has two phases:
      Phase 1 (Forward pass, Lines 5-11): Iterate through each generated token,
              narrow the interval [L^{(t)}, R^{(t)}) based on known tokens.
              Store L^{(t-1)}, Delta^{(t-1)} for each step.
      Phase 2 (Reverse rotation, Lines 12-15): Starting from t_end, compute
              mid^{(t)} and reverse-rotate back to mid^{(-1)}.
      Final (Lines 16-18): round_half_down(mid^{(-1)}) -> d_s -> m_s

    Args:
        idx: sample index
        context_text: original text context
        context: tokenized context tensor (1, seq_len)
        stegotext_tokens: tokenized full stegotext tensor (1, total_len), including context
        topk: top-k truncation size
        prng: random.Random instance seeded with key K (Algorithm 4 Line 2)

    Returns:
        extracted_bits: the extracted secret message bitstring, or None on failure
    """
    StartTime = time.time()

    # Algorithm 4, Line 3: [L^{(-1)}, R^{(-1)}) <- [0, 2^l)
    MAX_NUM = Decimal(2 ** args.bit_length)
    L = Decimal(0)
    R = Decimal(MAX_NUM)

    context_len = context.shape[1]
    total_len = stegotext_tokens.shape[1]

    # Algorithm 4, Line 4: t_end <- |S| - |C| - 1
    t_end = total_len - context_len - 1

    if t_end <= 0:
        print(f"[{idx}] Error: stegotext has no generated tokens beyond context.")
        return None

    # Storage for reverse rotation (Phase 2)
    # We need L^{(t-1)} and Delta^{(t-1)} for each t = 0, 1, ..., t_end
    L_history = []      # L^{(t-1)} at each step t
    Delta_history = []   # Delta^{(t-1)} at each step t
    o_history = []       # o^{(t)} at each step t

    # ============================================================
    # Phase 1: Forward pass with KV cache
    # Iterate t = 0, 1, ..., t_end to narrow the interval
    # ============================================================
    max_ctx_len = 1000
    input_ids = context[:, -max_ctx_len:]  # (1, seq_len)

    # Initial prefill: get KV cache and first-step logits
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
    past_key_values = outputs.past_key_values
    cached_len = input_ids.shape[1]

    # Algorithm 4, Line 5: for t = 0, 1, ..., t_end do
    # --- t = 0 (first token, from prefill logits) ---

    # Algorithm 4, Line 6: p^{(t)} <- M(S[:|C|+t])
    temp_logits, indices = outputs.logits[0, -1, :].to(torch.float64).sort(descending=True)
    probabilities = torch.nn.functional.softmax(temp_logits, dim=-1)

    # Record L^{(t-1)} and Delta^{(t-1)} before interval update
    L_history.append(L)
    Delta_history.append(R - L)

    # Generate and store o^{(t)} from PRNG (needed for reverse rotation)
    o = Decimal(prng.random())
    o_history.append(o)

    # Find which token in the top-k list matches the known stegotext token
    valid_probabilities = probabilities[:topk]
    valid_tokenlist = indices[:topk]

    # The known next token is S[|C| + 0 + 1] = S[context_len + 0]
    # (0-indexed: stegotext_tokens[0, context_len] is the first generated token)
    known_token_id = stegotext_tokens[0, context_len].item()
    token_pos = (valid_tokenlist == known_token_id).nonzero(as_tuple=True)[0]
    if len(token_pos) == 0:
        print(f"[{idx}] Error at t=0: token {known_token_id} not found in top-{topk} list.")
        return None
    known_token_index = token_pos[0].item()

    # Algorithm 4, Lines 10-11: Match known token and update interval
    L, R = range_decoding_detail(valid_probabilities, L, R, known_token_index)

    # Track the last token for incremental KV cache input
    next_token = valid_tokenlist[known_token_index]
    cached_len += 1

    # --- t = 1, 2, ..., t_end (subsequent tokens, incremental KV cache) ---
    for t in range(1, t_end + 1):
        # Incremental KV cache input
        new_input = next_token.unsqueeze(0).unsqueeze(0)  # (1, 1)

        # Trim KV cache if exceeding max context length
        if cached_len > max_ctx_len:
            trim = cached_len - max_ctx_len
            past_key_values = tuple(
                (k[:, :, trim:, :], v[:, :, trim:, :])
                for k, v in past_key_values
            )
            cached_len = max_ctx_len

        with torch.no_grad():
            outputs = model(new_input, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        cached_len += 1

        # Algorithm 4, Line 6: p^{(t)} <- M(S[:|C|+t])
        temp_logits, indices = outputs.logits[0, -1, :].to(torch.float64).sort(descending=True)
        probabilities = torch.nn.functional.softmax(temp_logits, dim=-1)

        # Record L^{(t-1)} and Delta^{(t-1)} before update
        L_history.append(L)
        Delta_history.append(R - L)

        # Generate and store o^{(t)}
        o = Decimal(prng.random())
        o_history.append(o)

        # Find matching token in top-k
        valid_probabilities = probabilities[:topk]
        valid_tokenlist = indices[:topk]

        known_token_id = stegotext_tokens[0, context_len + t].item()
        token_pos = (valid_tokenlist == known_token_id).nonzero(as_tuple=True)[0]
        if len(token_pos) == 0:
            print(f"[{idx}] Error at t={t}: token {known_token_id} not found in top-{topk} list.")
            return None
        known_token_index = token_pos[0].item()

        # Algorithm 4, Lines 10-11: Match known token and update interval
        L, R = range_decoding_detail(valid_probabilities, L, R, known_token_index)

        # Update next_token for KV cache
        next_token = valid_tokenlist[known_token_index]

    # ============================================================
    # Phase 2: Reverse rotation
    # Algorithm 4, Lines 12-15
    # ============================================================

    # Algorithm 4, Line 12: mid^{(t_end)} <- (L^{(t_end)} + R^{(t_end)}) / 2
    mid = (L + R) / 2

    # Algorithm 4, Line 13: for t = t_end, ..., 1, 0 do
    for t in range(t_end, -1, -1):
        # Algorithm 4, Line 14: o^{(t)} <- U(0,1).sample(PRNG(t))
        # (already stored in o_history[t])
        o_t = o_history[t]

        # Algorithm 4, Line 15: mid^{(t-1)} <- L^{(t-1)} + (mid^{(t)} - L^{(t-1)} - o^{(t)} * Delta^{(t-1)}) mod Delta^{(t-1)}
        L_prev = L_history[t]
        Delta_prev = Delta_history[t]
        mid = L_prev + (mid - L_prev - o_t * Delta_prev) % Delta_prev

    # ============================================================
    # Algorithm 4, Lines 16-18: Finalize extraction
    # ============================================================

    # Algorithm 4, Line 16: d_s^{(-1)} <- round_half_down(mid^{(-1)})
    # round_half_down: when exactly halfway (e.g., 2.5), round toward smaller value (e.g., 2)
    d_s = round_half_down(mid)

    # Algorithm 4, Line 17: m_s <- dec2bin(d_s^{(-1)}).zfill(l)
    extracted_bits = bin(int(d_s))[2:].zfill(args.bit_length)

    Duration = time.time() - StartTime
    return extracted_bits, Duration


def round_half_down(value):
    """
    Round half down: when a number is exactly halfway between two integers
    (e.g., 2.5), round toward the smaller integer (e.g., 2).
    This matches the round_half_down semantics in Algorithm 4, Line 16.
    """
    return value.to_integral_value(rounding=ROUND_HALF_DOWN)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--language_model', default='meta-llama/Llama-3.1-8B', type=str, required=False)
    parser.add_argument('--bit_length', default=128, type=int, required=False)
    parser.add_argument('--top_k', default=-1, type=int, required=False)
    parser.add_argument('--temp', default=1.0, type=float, required=False)
    # Algorithm 4, Line 2: Symmetric key K as PRNG seed
    parser.add_argument('--key', default=42, type=int, required=False,
                        help='Symmetric key K for PRNG seed (shared between Alice and Bob)')
    parser.add_argument('--input_file', default=None, type=str, required=True,
                        help='Path to the TSV file generated by encode_range_kvcache.py')
    args = parser.parse_args()
    print(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model_name = args.language_model
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, TOKENIZERS_PARALLELISM=True, padding_side='left')
    model.resize_token_embeddings(len(tokenizer.vocab))
    model.eval()

    # Read the encoded stegotext results
    input_df = pd.read_csv(args.input_file, sep='\t')

    success_count = 0
    fail_count = 0

    for index, row in input_df.iterrows():
        idx = row['Idx']
        context_text = row['Context']
        stegotext = row['Text']

        # Algorithm 4, Line 1: Tokenize t_s into S
        stego_tokens = tokenizer(stegotext, return_tensors="pt").to(model.device)['input_ids']
        context_tokens = tokenizer(context_text, return_tensors="pt").to(model.device)['input_ids']

        # Algorithm 4, Line 2: PRNG.set_seed(K)
        # Use an independent random.Random instance to avoid polluting global random state
        prng = random.Random(args.key)

        result = decode_range(
            idx, context_text, context_tokens, stego_tokens,
            topk=len(tokenizer.vocab) if args.top_k < 0 else args.top_k,
            prng=prng
        )

        if result is not None:
            extracted_bits, duration = result
            print(f"[{idx}] Extracted: {extracted_bits} (time: {duration:.4f}s)")
            success_count += 1
        else:
            print(f"[{idx}] Extraction failed.")
            fail_count += 1

        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nExtraction complete. Success: {success_count}, Failed: {fail_count}")
