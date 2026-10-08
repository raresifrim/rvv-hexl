// Copyright (C) 2020 Intel Corporation
// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

#include "hexl/util/defines.hpp"

namespace intel {
namespace hexl {

/// @brief Abstract memory-allocation strategy. The NTT class and
/// AlignedAllocator allocate all their tables through one of these, so a caller
/// can plug in its own memory pool (SEAL does). Interface only; nothing to port.
struct AllocatorBase {
	virtual ~AllocatorBase() noexcept {}

	/// @brief Allocates bytes_count bytes; returns nullptr on failure
	virtual void* allocate(size_t bytes_count) = 0;

	/// @brief Frees memory returned by allocate(); n is the size originally asked
	virtual void deallocate(void* p, size_t n) = 0;
};

/// @brief CRTP helper: implement allocate_impl / deallocate_impl in the derived
/// class and get the virtual AllocatorBase interface for free.
template <class AllocatorImpl>
struct AllocatorInterface : public AllocatorBase {
	void* allocate(size_t bytes_count) override {
		return static_cast<AllocatorImpl*>(this)->allocate_impl(bytes_count);
	}

	void deallocate(void* p, size_t n) override {
		static_cast<AllocatorImpl*>(this)->deallocate_impl(p, n);
	}

	private:
	// in case AllocatorImpl doesn't provide implementations, use default null
	// behavior
	void* allocate_impl(size_t bytes_count) {
		HEXL_UNUSED(bytes_count);
		return nullptr;
	}
	void deallocate_impl(void* p, size_t n) {
		HEXL_UNUSED(p);
		HEXL_UNUSED(n);
	}
};

}  // namespace hexl
}  // namespace intel
