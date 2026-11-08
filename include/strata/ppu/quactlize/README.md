# Shared quactlize ABI

These three headers are verbatim copies from the user's local
`llama.cpp/ggml/src/ggml-cuda/quactlize/` on 2026-11-06. That checkout's HEAD was
`2f288debe17e4acd261dcb86961f3a93ce8a26f2`, with existing local changes; HEAD alone
does not identify the additive APIs. `ABI_SHA256` pins the actual header bytes.

Run `sha256sum -c ABI_SHA256` in this directory. Strata builds no quactlize source
and changes no shared library. It loads the independent GPU producer
`libquactlize_ppu_pack.so` and the SIMT consumer
`libquactlize_ppu_execution.so` with `RTLD_LOCAL`.

Producer-owned arrangement-v2 descriptors and size queries are checked against
the consumer before allocation or enqueue. No Xplane descriptor, format-specific
legacy consumer, or ScaleFirst library is substituted. Q8_0 is admitted only
through its separate canonical map. See `docs/PPU.md` for execution and memory
limits; this integration does not bind the TC dispatcher or its tuning policy.
