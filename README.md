# llama-prism · low-bit KV cache (Ampere)

**A [PrismML/llama.cpp](https://github.com/PrismML-Eng/llama.cpp) fork that unlocks five extra low-bit KV-cache types on NVIDIA Ampere (RTX 30-series)** — fit a much larger context on a single 24 GB card, at essentially full prefill speed.

This is the Bonsai / PrismML `prism` line. Upstream flash-attention only ships fast matched KV kernels for `f16`/`q8_0`/`q4_0`; this fork adds **`q4_1`, `q5_0`, `q5_1`, `q3_K` and `q2_K`** as selectable `--cache-type-k` / `--cache-type-v` options.

| KV type | bits/elem | KV VRAM vs q8_0 | prefill pp2048 | decode tg32 | added by fork |
|--------:|:---------:|:---------------:|:--------------:|:-----------:|:---:|
| `q8_0` | 8.5 | — | 1610 t/s | 47.2 t/s | (upstream) |
| `q4_0` | 4.5 | ~47% smaller | 1586 t/s | 46.4 t/s | (upstream) |
| **`q5_1`** | 6.0 | ~29% smaller | 1584 t/s | 46.5 t/s | ✅ |
| **`q5_0`** | 5.5 | ~35% smaller | 1591 t/s | 45.7 t/s | ✅ |
| **`q4_1`** | 5.0 | ~41% smaller | 1589 t/s | 46.0 t/s | ✅ |
| **`q3_K`** | ~3.4 | **~60% smaller** | 1560 t/s | 42.2 t/s | ✅ |
| **`q2_K`** | ~2.6 | **~69% smaller** | 1547 t/s | 40.6 t/s | ✅ |

<sub>RTX 3090 (sm86, CUDA 13), Bonsai-2-27B PTQ1_0, flash-attention on, matched `K==V`. Mixing different K/V types disables flash-attention (upstream limitation) — always set K and V to the **same** type.</sub>

**Why it's fast:** on matched `K==V` the KV tensors are dequantized to f16 and run the existing fast f16 tensor-core kernels (MMA prefill / VEC decode), so prefill speed stays ~q8_0 while the cache shrinks. `q3_K` is the practical near-lossless choice for maximum context; `q2_K` is an experimental extreme-VRAM option (clamped super-block scale; usable at moderate context, too aggressive for 256K).

### Use it

```bash
# fits a much larger context in the same VRAM
llama-server -m model.gguf -ngl 99 -fa 1 -c 262144 -ctk q3_K -ctv q3_K
```

Requires `n_embd_k_gqa % 256 == 0` for `q3_K`/`q2_K`. **Prebuilt Windows CUDA 13 / Ampere (sm86) binaries** are on the [Releases page](https://github.com/Cybertiron/llama-prism-ampere-kv/releases) — download, unzip, run. To build from source, follow the standard llama.cpp CUDA build below.

### Credits

- `q3_K` / `q2_K` k-quant formats: **@ikawrakow** (llama.cpp k-quants).
- Base fork and ternary Bonsai runtime: **[PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp)**.
- Ampere KV-cache enabling/integration: this fork.

### About / support

Built by Cybertiron. If it saved you some VRAM:

<a href="https://www.buymeacoffee.com/cybertiron"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="46"></a>

---

## Built on llama.cpp

This repository is a fork of [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp) (the Bonsai / `prism` line, itself a fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)). Only the KV-cache additions described above are specific to this fork; everything else tracks upstream. For general build instructions, supported backends, model files, and usage, see the [upstream README](https://github.com/ggml-org/llama.cpp#readme) and the [build guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md).
