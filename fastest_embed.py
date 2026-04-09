import torch
import torch.nn.functional as F
import argparse
import numpy as np
import bitarray
import sys
import re
import math
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import random
import time
import os
import csv
from datasets import load_dataset
from itertools import islice
from decimal import *
import pandas as pd
import gc
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

getcontext().prec = 50


def range_coding_detail(valid_probabilities, current_min, current_max, secret_value):
    cumulative = torch.cumsum(valid_probabilities, dim=0)
    cumulative = cumulative / cumulative[-1]
    assert cumulative[-1].item() == 1.0
    boundries = torch.cat([torch.tensor([0.0], device=valid_probabilities.device), cumulative[:]])
    boundries_np = boundries.cpu().numpy()
    Max_interval = current_max - current_min
    selected_index = 'error'
    for i in range(len(boundries_np)):
        if current_min + Decimal(str(boundries_np[i+1])) * Max_interval > secret_value:
            selected_index = i
            current_min_new = current_min + Decimal(str(boundries_np[i])) * Max_interval
            current_max_new = current_min + Decimal(str(boundries_np[i+1])) * Max_interval
            break

    assert selected_index != 'error'
    return selected_index, current_min_new, current_max_new


def encode_range(idx, context_text, context, topk):
    StartTime = time.time()
    MAX_NUM = Decimal(2 ** args.bit_length)
    secret_value = Decimal(int(secret_bits, 2))
    token_total = 0
    kl_total = 0
    entropy_total = 0
    current_min = Decimal(0)
    current_max = Decimal(MAX_NUM)
    secret_value_temp = secret_value

    past_key_values = None

    # Prefill: run the full context through the model to build initial KV cache.
    # Truncate to the last 1000 tokens to stay within the model's context window.
    prefill_input = context[:, -1000:]
    with torch.no_grad():
        outputs = model(prefill_input, use_cache=True)
    past_key_values = outputs.past_key_values

    # Extract logits for the last position from the prefill pass
    temp_logits, indices = outputs.logits[0, -1, :].to(torch.float64).sort(descending=True)
    probabilities = torch.nn.functional.softmax(temp_logits, dim=-1)
    entropy_total += -torch.sum(probabilities * torch.log2(probabilities + 1e-10))

    offset = random.random()
    secret_value_temp = current_min + ((secret_value_temp - current_min) + (current_max - current_min) * Decimal(offset)) % (current_max - current_min)
    assert secret_value_temp >= current_min and secret_value_temp < current_max

    valid_probabilities = probabilities[:topk]
    valid_tokenlist = indices[:topk]
    selected_index, current_min, current_max = range_coding_detail(valid_probabilities, current_min, current_max, secret_value_temp)
    next_token = valid_tokenlist[selected_index]

    context = torch.cat((context, next_token.unsqueeze(0).unsqueeze(0)), dim=-1)
    token_total += 1

    # Check convergence after the first token
    if abs((current_min + current_max) / 2 - secret_value_temp) < 0.5:
        Duration = time.time() - StartTime
        bpt = args.bit_length / token_total
        avg_KL = kl_total / token_total
        avg_entropy = entropy_total / token_total
        Utilization = bpt / avg_entropy
        Stegotext = tokenizer.decode(context.squeeze(0), skip_special_tokens=True)
        with open(file_name, 'a+', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow([idx, avg_KL, bpt, avg_entropy.item(), Utilization.item(), Duration, '-', context_text, Stegotext])
        return True

    # Incremental decoding: feed only the latest token and reuse past_key_values
    while True:
        if token_total >= 2 * args.bit_length:
            return False

        # Only pass the single new token; the KV cache already holds all prior context
        new_input = next_token.unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            outputs = model(new_input, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values

        temp_logits, indices = outputs.logits[0, -1, :].to(torch.float64).sort(descending=True)
        probabilities = torch.nn.functional.softmax(temp_logits, dim=-1)
        entropy_total += -torch.sum(probabilities * torch.log2(probabilities + 1e-10))

        offset = random.random()
        secret_value_temp = current_min + ((secret_value_temp - current_min) + (current_max - current_min) * Decimal(offset)) % (current_max - current_min)
        assert secret_value_temp >= current_min and secret_value_temp < current_max

        valid_probabilities = probabilities[:topk]
        valid_tokenlist = indices[:topk]
        selected_index, current_min, current_max = range_coding_detail(valid_probabilities, current_min, current_max, secret_value_temp)
        next_token = valid_tokenlist[selected_index]

        context = torch.cat((context, next_token.unsqueeze(0).unsqueeze(0)), dim=-1)
        token_total += 1

        if abs((current_min + current_max) / 2 - secret_value_temp) < 0.5:
            break

    Duration = time.time() - StartTime
    bpt = args.bit_length / token_total
    avg_KL = kl_total / token_total
    avg_entropy = entropy_total / token_total
    Utilization = bpt / avg_entropy
    Stegotext = tokenizer.decode(context.squeeze(0), skip_special_tokens=True)
    with open(file_name, 'a+', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow([idx, avg_KL, bpt, avg_entropy.item(), Utilization.item(), Duration, '-', context_text, Stegotext])
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--language_model', default='openai-community/gpt2', type=str, required=False)
    parser.add_argument('--bit_length', default=128, type=int, required=False)
    parser.add_argument('--top_k', default=-1, type=int, required=False)
    parser.add_argument('--temp', default=1.0, type=float, required=False)
    parser.add_argument('--part', default=0, type=int, required=False)
    parser.add_argument('--part_max', default=2, type=int, required=False)
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
    file_name = '1.test_RC_decimal_' + model_name.rsplit('/', 1)[-1] + '_bit' + str(args.bit_length) + '.tsv'
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
            if True == encode_range(row['idx'], context, context_tokens, topk=len(tokenizer.vocab) if args.top_k < 0 else args.top_k):
                break
            gc.collect()
            torch.cuda.empty_cache()
