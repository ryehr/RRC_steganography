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
# Algorithm 3, Lines 6-8, 11-12: Interval construction & token selection & interval update
# ============================================================
def range_coding_detail(valid_probabilities, current_min, current_max, secret_value):
    """
    Corresponds to Algorithm 3:
      Line 6: c^{(t)} <- 0 || p^{(t)}.cumsum()
      Line 7: Delta^{(t-1)} = R^{(t-1)} - L^{(t-1)}
      Line 8: c'^{(t)} <- L^{(t-1)} + Delta^{(t-1)} * c^{(t)}
      Line 11: Select token_i so that d_s^{(t)} in [c'^{(t)}[i-1], c'^{(t)}[i])
      Line 12: [L^{(t)}, R^{(t)}) <- [c'^{(t)}[i-1], c'^{(t)}[i])
    """
    # Line 6: c^{(t)} <- 0 || p^{(t)}.cumsum()
    cumulative = torch.cumsum(valid_probabilities, dim=0)
    cumulative = cumulative / cumulative[-1]
    assert cumulative[-1].item() == 1.0
    boundries = torch.cat([torch.tensor([0.0], device=valid_probabilities.device), cumulative[:]])
    boundries_np = boundries.cpu().numpy()

    # Line 7: Delta^{(t-1)} = R^{(t-1)} - L^{(t-1)}
    Delta = current_max - current_min

    # Line 8 & 11 & 12:
    selected_index = 'error'
    for i in range(len(boundries_np) - 1):
        # c'^{(t)}[i+1] = L^{(t-1)} + Delta^{(t-1)} * c^{(t)}[i+1]
        upper = current_min + Decimal(str(boundries_np[i + 1])) * Delta
        if upper > secret_value:
            selected_index = i
            # Line 12: [L^{(t)}, R^{(t)})
            current_min_new = current_min + Decimal(str(boundries_np[i])) * Delta
            current_max_new = upper
            break

    assert selected_index != 'error'
    return selected_index, current_min_new, current_max_new


# ============================================================
# Termination condition check, corresponds to Algorithm 3, Lines 14-15
# Paper: (L^{(t)} + R^{(t)})/2 - d_s^{(t)} in (-0.5, 0.5]
# i.e., diff > -0.5 and diff <= 0.5
# ============================================================
def check_termination(current_min, current_max, d_s):
    """
    Algorithm 3, Lines 14-15:
      if (L^{(t)} + R^{(t)})/2 - d_s^{(t)} in (-0.5, 0.5] then break
    """
    diff = (current_min + current_max) / 2 - d_s
    return diff > Decimal('-0.5') and diff <= Decimal('0.5')


# ============================================================
# Algorithm 3: Rotation RC steganography (embed)
# ============================================================
def encode_range(idx, context_text, context, topk, prng):
    """
    Full embedding procedure corresponding to Algorithm 3, with KV cache optimization.

    Args:
        idx: sample index
        context_text: original text context
        context: tokenized context tensor (1, seq_len)
        topk: top-k truncation size
        prng: random.Random instance seeded with key K (Algorithm 3 Line 2)
    """
    StartTime = time.time()

    # Algorithm 3, Line 1: d_s^{(-1)} <- bin2dec(m_s)
    MAX_NUM = Decimal(2 ** args.bit_length)
    d_s = Decimal(int(secret_bits, 2))

    token_total = 0
    kl_total = 0
    entropy_total = 0

    # Algorithm 3, Line 3: [L^{(-1)}, R^{(-1)}) <- [0, 2^l)
    L = Decimal(0)
    R = Decimal(MAX_NUM)

    # ============================================================
    # KV Cache optimization: prefill context once, then input 1 new token per step
    # ============================================================
    max_ctx_len = 1000
    input_ids = context[:, -max_ctx_len:]  # (1, seq_len)

    # Initial prefill: get KV cache and first-step logits
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
    past_key_values = outputs.past_key_values
    cached_len = input_ids.shape[1]

    # Algorithm 3, Line 4: for t = 0, 1, ... do
    # --- t = 0 (first token, from prefill logits) ---

    # Algorithm 3, Line 5: p^{(t)} <- M(C)
    temp_logits, indices = outputs.logits[0, -1, :].to(torch.float64).sort(descending=True)
    probabilities = torch.nn.functional.softmax(temp_logits, dim=-1)
    entropy_total += -torch.sum(probabilities * torch.log2(probabilities + 1e-10))

    # Algorithm 3, Line 9: o^{(t)} <- U(0,1).sample(PRNG(t))
    o = Decimal(prng.random())

    # Algorithm 3, Line 10: d_s^{(t)} <- L^{(t-1)} + (d_s^{(t-1)} - L^{(t-1)} + o^{(t)} * Delta^{(t-1)}) mod Delta^{(t-1)}
    Delta = R - L
    d_s = L + ((d_s - L) + o * Delta) % Delta
    assert d_s >= L and d_s < R

    # Algorithm 3, Lines 6-8, 11-12: Interval construction + token selection + interval update
    valid_probabilities = probabilities[:topk]
    valid_tokenlist = indices[:topk]
    selected_index, L, R = range_coding_detail(valid_probabilities, L, R, d_s)

    # Algorithm 3, Line 13: C <- C || token_i
    next_token = valid_tokenlist[selected_index]
    all_token_ids = torch.cat((context, next_token.unsqueeze(0).unsqueeze(0)), dim=-1)
    token_total += 1
    cached_len += 1

    # Algorithm 3, Lines 14-15: Termination condition check
    if check_termination(L, R, d_s):
        Duration = time.time() - StartTime
        bpt = args.bit_length / token_total
        avg_KL = kl_total / token_total
        avg_entropy = entropy_total / token_total
        Utilization = bpt / avg_entropy
        Stegotext = tokenizer.decode(all_token_ids.squeeze(0), skip_special_tokens=True)
        with open(file_name, 'a+', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow([idx, avg_KL, bpt, avg_entropy.item(), Utilization.item(), Duration, '-', context_text, Stegotext])
        return True

    # --- t = 1, 2, ... (subsequent tokens, reuse KV cache with 1 token input per step) ---
    while True:
        if token_total >= 2 * args.bit_length:
            return False

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

        # Algorithm 3, Line 5: p^{(t)} <- M(C)
        temp_logits, indices = outputs.logits[0, -1, :].to(torch.float64).sort(descending=True)
        probabilities = torch.nn.functional.softmax(temp_logits, dim=-1)
        entropy_total += -torch.sum(probabilities * torch.log2(probabilities + 1e-10))

        # Algorithm 3, Line 9: o^{(t)} <- U(0,1).sample(PRNG(t))
        o = Decimal(prng.random())

        # Algorithm 3, Line 10: Rotate
        Delta = R - L
        d_s = L + ((d_s - L) + o * Delta) % Delta
        assert d_s >= L and d_s < R

        # Algorithm 3, Lines 6-8, 11-12
        valid_probabilities = probabilities[:topk]
        valid_tokenlist = indices[:topk]
        selected_index, L, R = range_coding_detail(valid_probabilities, L, R, d_s)

        # Algorithm 3, Line 13: C <- C || token_i
        next_token = valid_tokenlist[selected_index]
        all_token_ids = torch.cat((all_token_ids, next_token.unsqueeze(0).unsqueeze(0)), dim=-1)
        token_total += 1

        # Algorithm 3, Lines 14-15: Termination condition
        if check_termination(L, R, d_s):
            break

    # Algorithm 3, Lines 16-17: Detokenize C into t_s; return t_s
    Duration = time.time() - StartTime
    bpt = args.bit_length / token_total
    avg_KL = kl_total / token_total
    avg_entropy = entropy_total / token_total
    Utilization = bpt / avg_entropy
    Stegotext = tokenizer.decode(all_token_ids.squeeze(0), skip_special_tokens=True)
    with open(file_name, 'a+', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow([idx, avg_KL, bpt, avg_entropy.item(), Utilization.item(), Duration, '-', context_text, Stegotext])
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--language_model', default='meta-llama/Llama-3.1-8B', type=str, required=False)
    parser.add_argument('--bit_length', default=128, type=int, required=False)
    parser.add_argument('--top_k', default=-1, type=int, required=False)
    parser.add_argument('--temp', default=1.0, type=float, required=False)
    parser.add_argument('--part', default=0, type=int, required=False)
    parser.add_argument('--part_max', default=2, type=int, required=False)
    # Algorithm 3, Line 2: Symmetric key K as PRNG seed
    parser.add_argument('--key', default=42, type=int, required=False,
                        help='Symmetric key K for PRNG seed (shared between Alice and Bob)')
    args = parser.parse_args()
    print(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    model_name = args.language_model
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, TOKENIZERS_PARALLELISM=True, padding_side='left')
    model.resize_token_embeddings(len(tokenizer.vocab))
    model.eval()
    prompt_df = pd.read_csv('0.Prompts.tsv', sep='\t')
    header = ['Idx', 'KL', 'BPT', 'Entropy', 'Utilization', 'Time', 'PPL', 'Context', 'Text']
    file_name = '1.RC_decimal_' + model_name.rsplit('/', 1)[-1] + '_bit' + str(args.bit_length) + '.tsv'
    if os.path.exists(file_name) == False:
        with open(file_name, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(header)
    KL_list = []
    bpt_list = []
    entropy_list = []
    utilization_list = []
    time_list = []
    for index, row in prompt_df.iterrows():
        if row['idx'] < args.part * len(prompt_df) / args.part_max or row['idx'] >= (args.part + 1) * len(prompt_df) / args.part_max:
            continue
        while True:
            secret_bits = ''.join(str(random.randint(0, 1)) for _ in range(args.bit_length))
            context = row['text']
            context_tokens = tokenizer(context, return_tensors="pt").to(model.device)['input_ids']

            # Algorithm 3, Line 2: PRNG.set_seed(K)
            # Use an independent random.Random instance to avoid polluting global random state
            prng = random.Random(args.key)

            if True == encode_range(row['idx'], context, context_tokens,
                                    topk=len(tokenizer.vocab) if args.top_k < 0 else args.top_k,
                                    prng=prng):
                break
            gc.collect()
            torch.cuda.empty_cache()
