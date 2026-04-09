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

getcontext().prec = 50


def range_coding_detail(valid_probabilities, current_min, current_max, secret_value):
    cumulative = torch.cumsum(valid_probabilities, dim=0)
    cumulative = cumulative / cumulative[-1]
    assert cumulative[-1].item() == 1.0
    boundries = torch.cat([torch.tensor([0.0], device=valid_probabilities.device), cumulative[:]])
    boundries_np = boundries.cpu().numpy()
    Max_interval = current_max - current_min
    selected_index = 'error'
    for i in range(len(boundries_np) - 1):
        if current_min + Decimal(str(boundries_np[i + 1])) * Max_interval > secret_value:
            selected_index = i
            current_min_new = current_min + Decimal(str(boundries_np[i])) * Max_interval
            current_max_new = current_min + Decimal(str(boundries_np[i + 1])) * Max_interval
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

    # ============================================================
    # KV Cache 优化：首次 prefill context，之后每步只输入 1 个新 token
    # ============================================================

    # 截断 context 到最大长度（与原逻辑一致）
    max_ctx_len = 1000
    input_ids = context[:, -max_ctx_len:]  # (1, seq_len)

    # 第一次 forward：prefill 整个 context，获取初始 KV cache
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
    past_key_values = outputs.past_key_values
    # 记录当前 KV cache 中已缓存的 token 数量
    cached_len = input_ids.shape[1]

    # 从 prefill 的最后一个位置取 logits（预测下一个 token）
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

    # 把新 token 拼到完整序列（用于最终 decode）
    all_token_ids = torch.cat((context, next_token.unsqueeze(0).unsqueeze(0)), dim=-1)
    token_total += 1
    cached_len += 1

    # 检查首个 token 后是否已经满足终止条件
    if abs((current_min + current_max) / 2 - secret_value_temp) < 0.5:
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

    # ============================================================
    # 自回归循环：每步只输入 1 个 token，复用 past_key_values
    # ============================================================
    while True:
        if token_total >= 2 * args.bit_length:
            return False

        # 只输入刚生成的 1 个 token + 传入 past_key_values
        new_input = next_token.unsqueeze(0).unsqueeze(0)  # (1, 1)

        # 如果 KV cache 超过 max_ctx_len，需要裁剪
        # past_key_values 的每一层形状: (batch, num_heads, seq_len, head_dim)
        if cached_len > max_ctx_len:
            # 裁掉最旧的 token（保留最近 max_ctx_len - 1 个，加上即将输入的 1 个 = max_ctx_len）
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

        all_token_ids = torch.cat((all_token_ids, next_token.unsqueeze(0).unsqueeze(0)), dim=-1)
        token_total += 1

        if abs((current_min + current_max) / 2 - secret_value_temp) < 0.5:
            break

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
            if True == encode_range(row['idx'], context, context_tokens, topk=len(tokenizer.vocab) if args.top_k < 0 else args.top_k):
                break
            gc.collect()
            torch.cuda.empty_cache()
