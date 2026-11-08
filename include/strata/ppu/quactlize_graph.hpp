#pragma once

namespace strata::ppu::quactlize {
// Graph capture cannot allocate, and an offline packed expert layer has no raw GPU reader to fall back to.
// Reserve, before capture, the scratch of the grouped() calls that will first run inside a captured graph:
// keyed like grouped() by `out` and the call kind (gate/up or down), sized for cap_entries. False: grouped()
// would decline this shape too (or the STRATA_QUACTLIZE_SCRATCH_MIB budget is exhausted). Throws in capture.
bool prepare_grouped(float* out, bool gate_up, int type, int n, int k, int cap_entries, void* stream,
                     bool packed = false);
} // namespace strata::ppu::quactlize
