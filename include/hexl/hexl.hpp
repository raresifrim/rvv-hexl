// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// Umbrella header, as included by OpenFHE ("hexl/hexl.hpp").
// Differences from upstream v1.2.6:
//   - hexl/experimental/* (SEAL key-switch, dyadic multiply, FFT-like, LR
//     mat-vec) is not ported: neither OpenFHE's HEXL backend nor SEAL's
//     core path needs it. Add it back here if you ever port it.
//   - hexl/rvv/port-info.hpp is added (build/runtime introspection).

#pragma once

#include "hexl/eltwise/eltwise-add-mod.hpp"
#include "hexl/eltwise/eltwise-cmp-add.hpp"
#include "hexl/eltwise/eltwise-cmp-sub-mod.hpp"
#include "hexl/eltwise/eltwise-fma-mod.hpp"
#include "hexl/eltwise/eltwise-mult-mod.hpp"
#include "hexl/eltwise/eltwise-reduce-mod.hpp"
#include "hexl/eltwise/eltwise-sub-mod.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/ntt/ntt.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "hexl/rvv/port-info.hpp"
#include "hexl/util/aligned-allocator.hpp"
#include "hexl/util/check.hpp"
#include "hexl/util/compiler.hpp"
#include "hexl/util/defines.hpp"
#include "hexl/util/types.hpp"
#include "hexl/util/util.hpp"
