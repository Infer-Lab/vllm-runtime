# vllm-runtime

vLLM built from source for NVIDIA Blackwell (SM120) and Hopper (SM90), with FlashInfer and DeepGEMM from source and SM120 fixes not yet upstream.

## Image

| Tag | Built | Digest |
|---|---|---|
| `ghcr.io/infer-lab/vllm-runtime:20261108-4ef66554bdd4` | 2026-11-08 | `sha256:5e8796e998239731fb6ede3c8fec348bc5d640349526f628f670d35a56511a75` |
| `ghcr.io/infer-lab/vllm-runtime:stable` | 2026-11-08 | `sha256:5e8796e998239731fb6ede3c8fec348bc5d640349526f628f670d35a56511a75` |

## Usage

The entrypoint activates the image's environment and runs the given command.

```sh
docker run --gpus all --ipc=host -p 8000:8000 -v /path/to/models:/models \
  ghcr.io/infer-lab/vllm-runtime:stable \
  vllm serve /models/<model> --tensor-parallel-size 4
```

## Contents

### vllm

[`ea8954910b3d`](https://github.com/lucifer1004/vllm/commit/ea8954910b3df55a228110fb73d49e33c874a4f2): [vllm-project/vllm](https://github.com/vllm-project/vllm) at [`73d0083d28`](https://github.com/vllm-project/vllm/commit/73d0083d28)

plus:

- [vllm#55264](https://github.com/vllm-project/vllm/pull/55264): [Bugfix][SM120] Scope sparse MLA warmup autotuning to attention
- [vllm#57351](https://github.com/vllm-project/vllm/pull/57351): [Bugfix][NVFP4 MoE] Pad gated intermediate for FlashInfer/CUTLASS
- [vllm#60069](https://github.com/vllm-project/vllm/pull/60069): [Bugfix][Kernel] Share FlashInfer b12x MoE workspaces across layers
- [vllm#58863](https://github.com/vllm-project/vllm/pull/58863): [Spec Decode][Qwen4Exp] RecoverSSM for Qwen GDN and the PLE short conv
- [vllm#59366](https://github.com/vllm-project/vllm/pull/59366): [Spec Decode][Qwen4Exp] Run GDN RecoverSSM verify in the fused CUDA MTP kernel
- [chunked-host-pin](https://github.com/lucifer1004/vllm/tree/fix/chunked-host-pin-73d0083): [Bugfix] Pin the Qwen4Exp PLE and DeepSeek-V4.1 Engram tables in 64 MiB chunks when one pin call is capped
- [vllm#59846](https://github.com/vllm-project/vllm/pull/59846): [Perf][DSv4.1] SM120: decode DeepSeek-V4.1's own KV records with FlashInfer's DSv4.1 sparse MLA
- [stop-strings-after-reasoning](https://github.com/lucifer1004/vllm/tree/fix/stop-strings-after-reasoning-ce49): [Bugfix][Frontend] Match stop strings only after reasoning ends
- [vllm#53078](https://github.com/vllm-project/vllm/pull/53078): [Bugfix][KV Connector] Mooncake: heterogeneous-TP support for hybrid GDN/Mamba models
- [vllm#59592](https://github.com/vllm-project/vllm/pull/59592): [Bugfix][KV Connector] Mooncake: reject mismatched block sizes in the handshake
- [vllm#59594](https://github.com/vllm-project/vllm/pull/59594): [Core][KV Connector] Keep the connector's head order when its KV layout is unsupported
- [vllm#58094](https://github.com/vllm-project/vllm/pull/58094): [Bugfix][Spec Decode] Propagate rope hf_overrides to MTP draft
- [sm120-flashinfer-dsa-indexer](https://github.com/lucifer1004/vllm/tree/feat/sm120-flashinfer-dsa-indexer): [Perf][SM120] Use FlashInfer's fused DSA indexer for FP8 prefill and decode

### flashinfer

[`757780dcb9d6`](https://github.com/lucifer1004/flashinfer/commit/757780dcb9d6c12d8254970461db82cbab791c67): [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) at [`5d2c80277`](https://github.com/flashinfer-ai/flashinfer/commit/5d2c80277)

plus:

- [flashinfer#8740](https://github.com/flashinfer-ai/flashinfer/pull/8740): feat(sm120_dsa_indexer): DeepSeek-V4.1 decode in the paged form: strided page caches and candidate blocks

### DeepGEMM-vllm

[`81960e53d003`](https://github.com/lucifer1004/DeepGEMM/commit/81960e53d0038ede0a28fe26862861665c612b75): [vllm-project/DeepGEMM](https://github.com/vllm-project/DeepGEMM) at [`82ca617`](https://github.com/vllm-project/DeepGEMM/commit/82ca617)

plus:

- [vllm-project/DeepGEMM#28](https://github.com/vllm-project/DeepGEMM/pull/28): [SM120] Support 128-state pages in FP8 paged-MQA logits
- [yyihuang/DeepGEMM#1](https://github.com/yyihuang/DeepGEMM/pull/1): [SM120] Extend larger paged-MQA pages to FP4 and 256-state pages

## Serving

Each model was gated with the command below, run on the image's entrypoint with the model directory mounted; the parallelism and every setting shown are the ones that were measured.

### [Qwen/Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) — TP1

```sh
vllm serve <model> --tensor-parallel-size 1 --max-model-len 16384 --gpu-memory-utilization 0.85 --trust-remote-code
```

### [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) — TP2

```sh
vllm serve <model> --language-model-only --speculative-config {"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic"} --use-replayssm --mamba-ssm-cache-dtype bfloat16 --tensor-parallel-size 2 --max-model-len 262144 --gpu-memory-utilization 0.9 --reasoning-parser qwen3 --enable-prompt-tokens-details
```

Environment: `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1`

### [nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) — TP4

```sh
vllm serve <model> --language-model-only --max-num-seqs 32 --max-num-batched-tokens 8192 --tensor-parallel-size 4 --max-model-len 65536 --kv-cache-dtype fp8 --gpu-memory-utilization 0.9 --tool-call-parser glm47 --reasoning-parser glm45 --enable-auto-tool-choice --enable-flashinfer-autotune
```

Environment: `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1`

### `deepseek-v4-1-flash` — TP4

```sh
vllm serve <model> --max-num-seqs 64 --max-num-batched-tokens 8192 --tensor-parallel-size 4 --max-model-len 65536 --kv-cache-dtype fp8 --gpu-memory-utilization 0.9 --block-size 64 --tokenizer-mode deepseek_v41 --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41 --enable-auto-tool-choice
```

## Validation

Gated on RTX PRO 6000 Blackwell (SM120) GPUs before publishing; scores are chat GSM8K (lm-eval, 1,319 questions, flexible-extract).

| Model | Configuration | Eval | Result |
|---|---|---|---|
| [Qwen/Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) | TP1 | smoke | passed |
| [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) | TP2 | smoke | passed |
| [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) | TP2 | gsm8k-chat | 0.9742 (gate 0.96) |
| [nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) | TP4 | smoke | passed |
| [nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) | TP4 | gsm8k-chat | 0.9719 (gate 0.955) |
| `deepseek-v4-1-flash` | TP4 | smoke | passed |
| `deepseek-v4-1-flash` | TP4 | gsm8k-chat | 0.9439 (gate 0.93) |

Built and gated with InferLab 0.23.3.
