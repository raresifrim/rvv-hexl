// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// Builders of the small-span twiddle layouts (layouts: ntt-rvv-small-span.hpp).
// Scalar gathers from the NTT tables, once per NTT object.

#include "ntt/ntt-rvv-small-span.hpp"

#ifdef HEXL_HAS_RVV

#include "util/rvv-util.hpp"

namespace intel {
namespace hexl {

template <class Strategy, typename E>
NttSmallSpanTables<Strategy, E> BuildNttSmallSpanTables(
		uint64_t n, uint64_t vl, const uint64_t* w, const uint64_t* w_precon,
		const uint64_t* w_inv, const uint64_t* w_inv_precon) {
	NttSmallSpanTables<Strategy, E> tw;
	tw.n = n;
	if (!SmallSpanRuns<Strategy>(n, vl)) return tw;  // vl = 0: the kernels run per block
	tw.vl = vl;

	// one entry: forward table index kf, or inverse table index ki
	auto fwd = [&](uint64_t kf) {
		tw.fwd.push_back(static_cast<E>(w[kf]));
		tw.fwd_precon.push_back(static_cast<E>(w_precon[kf]));
	};
	auto inv = [&](uint64_t ki) {
		tw.inv.push_back(static_cast<E>(w_inv[ki]));
		tw.inv_precon.push_back(static_cast<E>(w_inv_precon[ki]));
	};
	// stage t: m = n / (2t) blocks, forward twiddles at W[m + block],
	// inverse at Winv[n - 2m + 1 + block] = Winv[n - n/t + 1 + block]

	if constexpr (std::is_same_v<Strategy, rvv::ntt::FusedSlides>) {
		for (uint64_t c = 0; c < n / vl; c++) {                // chunk-major
			for (uint64_t t = vl / 2; t >= 1; t >>= 1)          // forward stage order
				for (uint64_t l = 0; l < vl; l++)
					fwd(n / (2 * t) + (c * vl + l) / (2 * t));
			for (uint64_t t = 1; t < vl; t <<= 1)                // inverse stage order
				for (uint64_t l = 0; l < vl; l++)
					inv(n - n / t + 1 + (c * vl + l) / (2 * t));
		}
	} else if constexpr (std::is_same_v<Strategy, rvv::ntt::Transposed>) {
		const uint64_t rows = n / vl;                          // >= vl
		for (uint64_t t = vl / 2; t >= 1; t >>= 1)              // forward stage order
			for (uint64_t g = 0; g < rows / (2 * t); g++)         // row groups
				for (uint64_t k = 0; k < vl; k++)                   // lane k: block k*R/(2t) + g
					fwd(n / (2 * t) + k * rows / (2 * t) + g);
		for (uint64_t t = 1; t < vl; t <<= 1)                   // inverse stage order
			for (uint64_t g = 0; g < rows / (2 * t); g++)
				for (uint64_t k = 0; k < vl; k++)
					inv(n - n / t + 1 + k * rows / (2 * t) + g);
	}
	return tw;
}

std::shared_ptr<const NttSmallSpanSet> BuildNttSmallSpanSet(const NTT& ntt) {
	using S32 = rvv::cfg::NttSmallSpan32;
	using S64 = rvv::cfg::NttSmallSpan64;
	const uint64_t n = ntt.GetDegree();
	auto set = std::make_shared<NttSmallSpanSet>();
	if (ntt.GetModulus() < NTT::s_max_fwd_32_modulus) {
		const uint64_t vl32 = rvv::VlMax<rvv::cfg::Ntt32>();
		const uint64_t vl32u64 = rvv::VlMax<rvv::cfg::Ntt32U64>();
		if (SmallSpanRuns<S32>(n, vl32))
			set->rvv32 = std::make_shared<const NttSmallSpanSet::T32>(BuildNttSmallSpanTables<S32, uint32_t>(ntt, vl32));
		if (vl32u64 == vl32)
			set->rvv32_u64 = set->rvv32;
		else if (SmallSpanRuns<S32>(n, vl32u64))
			set->rvv32_u64 = std::make_shared<const NttSmallSpanSet::T32>(BuildNttSmallSpanTables<S32, uint32_t>(ntt, vl32u64));
	} else {
		const uint64_t vl64 = rvv::VlMax<rvv::cfg::Ntt64>();
		if (SmallSpanRuns<S64>(n, vl64))
			set->rvv64 = std::make_shared<const NttSmallSpanSet::T64>(BuildNttSmallSpanTables<S64, uint64_t>(ntt, vl64));
	}
	if (!set->rvv32 && !set->rvv32_u64 && !set->rvv64) return nullptr;
	return set;
}

#define HEXL_SMALL_SPAN_TABLES(S, E)                                                   \
	template NttSmallSpanTables<S, E> BuildNttSmallSpanTables<S, E>(               \
			uint64_t, uint64_t, const uint64_t*, const uint64_t*, const uint64_t*, \
			const uint64_t*)
HEXL_SMALL_SPAN_TABLES(rvv::ntt::PerBlock, uint32_t);
HEXL_SMALL_SPAN_TABLES(rvv::ntt::PerBlock, uint64_t);
HEXL_SMALL_SPAN_TABLES(rvv::ntt::FusedSlides, uint32_t);
HEXL_SMALL_SPAN_TABLES(rvv::ntt::FusedSlides, uint64_t);
HEXL_SMALL_SPAN_TABLES(rvv::ntt::Transposed, uint32_t);
HEXL_SMALL_SPAN_TABLES(rvv::ntt::Transposed, uint64_t);
#undef HEXL_SMALL_SPAN_TABLES

}  // namespace hexl
}  // namespace intel

#endif  // HEXL_HAS_RVV
