#!/usr/bin/env python3
"""
GPU memory model: full-attention vs retrieval, as a function of context length.

Substantiates the core claim -- a 20GB GPU doing the work of a 128-256GB rig --
by computing the KV-cache that full attention must hold in VRAM versus the
constant footprint of the retrieval approach (model weights + a small retrieved
window; the corpus lives on RAM/disk).

    python3 -m groundwire.memory_model
    python3 -m groundwire.memory_model --model qwen2.5-72b --kv-dtype fp8

KV cache per token = 2 (K,V) * layers * kv_heads * head_dim * bytes_per_elem.
"""

from __future__ import annotations
import argparse

GB = 1024 ** 3

# (layers, kv_heads, head_dim, params_billions)
MODELS = {
    "qwen2.5-7b":  (28, 4, 128, 7.6),
    "llama3.1-8b": (32, 8, 128, 8.0),
    "qwen2.5-14b": (48, 8, 128, 14.8),
    "qwen2.5-72b": (80, 8, 128, 72.7),
    "llama3.1-70b": (80, 8, 128, 70.6),
}

DTYPE_BYTES = {"fp16": 2, "fp8": 1, "int4": 0.5}


def kv_bytes_per_token(layers, kv_heads, head_dim, kv_bytes):
    return 2 * layers * kv_heads * head_dim * kv_bytes


def weights_gb(params_b, weight_bytes):
    return params_b * 1e9 * weight_bytes / GB


def gpu_class(gb):
    if gb <= 24:
        return "1x 24GB (RTX 4090)"
    if gb <= 48:
        return "1x 48GB (A6000)"
    if gb <= 80:
        return "1x 80GB (A100/H100)"
    if gb <= 160:
        return "2x 80GB"
    if gb <= 320:
        return "4x 80GB (~256GB)"
    return f"{int(gb // 80) + 1}x 80GB (multi-node)"


def run(model, contexts, kv_dtype, weight_dtype, k_chunks, chunk_tokens, prompt):
    layers, kv_heads, head_dim, params = MODELS[model]
    kv_pt = kv_bytes_per_token(layers, kv_heads, head_dim, DTYPE_BYTES[kv_dtype])
    w_gb = weights_gb(params, DTYPE_BYTES[weight_dtype])

    # retrieval only ever puts (prompt + k retrieved chunks) through attention
    retr_tokens = prompt + k_chunks * chunk_tokens
    retr_kv_gb = retr_tokens * kv_pt / GB
    retr_total = w_gb + retr_kv_gb + 1.0  # +activations/overhead

    print(f"\nModel: {model}  | weights {w_gb:.1f}GB ({weight_dtype}) | "
          f"KV {kv_pt/1024:.0f} KB/token ({kv_dtype})")
    print(f"Retrieval always processes ~{retr_tokens:,} tokens "
          f"(prompt {prompt} + {k_chunks}x{chunk_tokens} chunks)\n")
    print(f"{'context':>10} {'full-attn VRAM':>16} {'GPU needed':>22} "
          f"{'retrieval VRAM':>15} {'ratio':>7}")
    print("-" * 76)
    rows = []
    for ctx in contexts:
        full_kv_gb = ctx * kv_pt / GB
        full_total = w_gb + full_kv_gb + 1.0
        ratio = full_total / retr_total
        rows.append((ctx, full_total, retr_total, ratio))
        print(f"{ctx:>10,} {full_total:>14.1f}GB {gpu_class(full_total):>22} "
              f"{retr_total:>13.1f}GB {ratio:>6.1f}x")
    print()
    biggest = rows[-1]
    print(f"At {biggest[0]:,} tokens: full attention needs ~{biggest[1]:.0f}GB "
          f"({gpu_class(biggest[1])}); retrieval needs ~{biggest[2]:.0f}GB "
          f"-> {biggest[3]:.0f}x less GPU memory, same model.")
    print("The corpus itself lives on RAM/disk (see the harness); the GPU never "
          "holds it.\n")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen2.5-7b", choices=list(MODELS))
    ap.add_argument("--contexts", type=int, nargs="+",
                    default=[64000, 256000, 1000000, 2000000, 3000000])
    ap.add_argument("--kv-dtype", default="fp16", choices=list(DTYPE_BYTES))
    ap.add_argument("--weight-dtype", default="fp16", choices=list(DTYPE_BYTES))
    ap.add_argument("--k-chunks", type=int, default=5)
    ap.add_argument("--chunk-tokens", type=int, default=450)
    ap.add_argument("--prompt", type=int, default=2000)
    args = ap.parse_args()
    run(args.model, args.contexts, args.kv_dtype, args.weight_dtype,
        args.k_chunks, args.chunk_tokens, args.prompt)


if __name__ == "__main__":
    main()
