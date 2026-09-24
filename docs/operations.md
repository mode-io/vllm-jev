# Operations

Choose free GPUs before startup; do not place a second 90%-memory vLLM instance on an occupied card. Keep the runtime, checkpoint and caches on an appropriately sized data volume. `scripts/serve.sh` resolves its workspace from the repository directory unless `VLLM_JEV_WORKSPACE` is set.

```bash
export CUDA_VISIBLE_DEVICES=0
export VLLM_JEV_CHECKPOINT=/path/to/jev-checkpoint
export VLLM_JEV_PORT=8795
bash scripts/serve.sh
```

The script validates checkpoint checksums, enables only the Jev model and endpoint plugins, and calls `vllm serve` on loopback. It defaults to compiled, asynchronous scheduling, hybrid prefix caching and `VLLM_JEV_GPU_MEMORY_UTILIZATION=0.9`. Set `VLLM_JEV_KV_BYTES` to a positive byte count only when fixed KV allocation is needed; it overrides GPU utilization. `VLLM_JEV_EAGER=1` and `VLLM_JEV_ASYNC=0` are diagnostic alternatives. Extra native vLLM flags are passed through unchanged:

```bash
bash scripts/serve.sh --max-num-seqs 64 --max-num-batched-tokens 8192
```

Check `/health`, then POST a Choice request and inspect `/metrics` for running/waiting candidate requests, KV utilization, preemptions and prefix-cache hits. `cached_tokens` in each Jev response measures actual reusable tokens; idle allocation of KV memory does not imply cache hits. The response's `confidence` is a uniform-adjusted score, not an empirical probability of correctness.

For multiple GPUs, use one model replica per card and route complete Jev decisions through `scripts/router.py`:

```bash
runtime/bin/python scripts/router.py \
  --upstream http://127.0.0.1:8795 \
  --upstream http://127.0.0.1:8796 --port 8793
```

The router keeps explicit `cache_salt` values on a stable replica and retries transport failures. Native `vllm serve --data-parallel-size 2` is also supported, but its per-candidate dispatch may reduce within-decision shared-prefix reuse. See the [feature map](vllm_features.md).

The optional gRPC investigation and load/stability scripts are deliberately stored outside this source repository. GitHub Actions is not used for this repository. Keep the HTTP and any gRPC listener on loopback or behind an authenticating proxy.
