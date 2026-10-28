// Copyright (C) 2026 Rares Ifrim, National University of Science and Technology POLITEHNICA Bucharest
// All rights reserved.
//
// hexl/eltwise: all seven public functions, plus the rvv-hexl extensions
// EltwiseMultAddMod, SignedDigitDecompose, BaseConvert, EltwiseExternalProductMod and EltwiseDotProductMod, every size tail, every modulus class, every input/output
// mod factor, out-of-place AND in-place (OpenFHE calls almost everything with
// result == operand1).
//
// Every test is a template on the storage word. Eltwise_<Op> runs it with
// uint64_t (the upstream API; validated against Intel HEXL); Eltwise_<Op>_32
// runs the same checks with uint32_t (the rvv-hexl 32-bit extension, for
// OpenFHE NATIVE_SIZE=32). Only the modulus ranges differ, following each
// function's documented preconditions.

#include <type_traits>
#include <utility>
#include <vector>

#include "hexl/hexl.hpp"
#include "oracle.hpp"
#include "test.hpp"

using namespace intel::hexl;
namespace O = hexltest::oracle;

namespace {

template <typename Word>
constexpr bool Is32() {
	return std::is_same<Word, uint32_t>::value;
}

/// Candidate moduli: every size class, both sides of the e32/e64 dispatch
/// boundary (2^30) and of the 32-bit storage limit. Each test keeps the ones
/// its preconditions allow for the word under test.
std::vector<uint64_t> Candidates() {
	std::vector<uint64_t> qs = {2, 3, 17};
	for (size_t bits : {20, 27, 29}) qs.push_back(O::NttPrime(bits, 1));
	qs.push_back(O::PrimeBelow(1ULL << 30));  // last modulus on the e32 path
	qs.push_back(O::PrimeAbove(1ULL << 30));  // first one past it
	qs.push_back(O::NttPrime(31, 1));         // in [2^31, 2^32)
	qs.push_back(O::PrimeBelow(1ULL << 32));  // largest 32-bit modulus
	for (size_t bits : {32, 40, 49, 50, 52, 59, 60, 61, 62}) {
		qs.push_back(O::NttPrime(bits, 1));
	}
	qs.push_back(O::PrimeBelow(1ULL << 63));
	return qs;
}

/// Candidates q with pred(q) true.
template <class P>
std::vector<uint64_t> Moduli(P pred) {
	std::vector<uint64_t> out;
	for (uint64_t q : Candidates()) {
		if (pred(q)) out.push_back(q);
	}
	return out;
}

/// Runs f(result, operand) out-of-place (into a fresh buffer) and in-place
/// (into a copy of `src`), and checks both against `want`.
template <typename Word, class F>
void CheckBothWays(const std::vector<Word>& want, const std::vector<Word>& src,
		F f, const char* what, uint64_t q) {
	const size_t n = want.size();
	std::vector<Word> out(n, static_cast<Word>(0xdeadbeefULL));
	f(out.data(), static_cast<const Word*>(src.data()));
	CHECK_VEC_EQ(want, out, << what << " out-of-place n=" << n << " q=" << q
			<< " word=" << 8 * sizeof(Word));

	std::vector<Word> inplace = src;
	f(inplace.data(), static_cast<const Word*>(inplace.data()));
	CHECK_VEC_EQ(want, inplace, << what << " in-place n=" << n << " q=" << q
			<< " word=" << 8 * sizeof(Word));
}

bool Compare(CMPINT cmp, uint64_t a, uint64_t bound) {
	switch (cmp) {
		case CMPINT::EQ: return a == bound;
		case CMPINT::LT: return a < bound;
		case CMPINT::LE: return a <= bound;
		case CMPINT::FALSE: return false;
		case CMPINT::NE: return a != bound;
		case CMPINT::NLT: return a >= bound;
		case CMPINT::NLE: return a > bound;
		case CMPINT::TRUE: return true;
	}
	return false;
}

const CMPINT kCmps[] = {CMPINT::EQ,  CMPINT::LT,  CMPINT::LE,  CMPINT::FALSE,
	CMPINT::NE,  CMPINT::NLT, CMPINT::NLE, CMPINT::TRUE};

// ---------------------------------------------------------------------------

template <typename Word>
void RunAddSub(bool add) {
	// 64-bit: q < 2^63 (the sum fits). 32-bit: q <= 2^32 (values fit in Word).
	auto ok = [](uint64_t q) { return Is32<Word>() ? q <= (1ULL << 32) : q < (1ULL << 63); };
	for (uint64_t q : Moduli(ok)) {
		for (size_t n : O::EltwiseSizes()) {
			auto a64 = O::Random(n, q), b64 = O::Random(n, q);
			if (add) { a64[0] = q - 1; b64[0] = q - 1; }  // largest possible sum
			else     { a64[0] = 0;     b64[0] = q - 1; }  // largest borrow
			const uint64_t s = O::Random(1, q)[0];
			std::vector<Word> want_vv(n), want_vs(n);
			for (size_t i = 0; i < n; ++i) {
				want_vv[i] = static_cast<Word>(add ? O::AddMod(a64[i], b64[i], q) : O::SubMod(a64[i], b64[i], q));
				want_vs[i] = static_cast<Word>(add ? O::AddMod(a64[i], s, q) : O::SubMod(a64[i], s, q));
			}
			const auto a = O::As<Word>(a64), b = O::As<Word>(b64);
			if (add) {
				CheckBothWays(want_vv, a, [&](Word* r, const Word* x) {
						EltwiseAddMod(r, x, b.data(), n, q); }, "AddMod vv", q);
				CheckBothWays(want_vs, a, [&](Word* r, const Word* x) {
						EltwiseAddMod(r, x, s, n, q); }, "AddMod vs", q);
			} else {
				CheckBothWays(want_vv, a, [&](Word* r, const Word* x) {
						EltwiseSubMod(r, x, b.data(), n, q); }, "SubMod vv", q);
				CheckBothWays(want_vs, a, [&](Word* r, const Word* x) {
						EltwiseSubMod(r, x, s, n, q); }, "SubMod vs", q);
			}
		}
	}
}

template <typename Word>
void RunMultMod() {
	for (uint64_t imf : {1ULL, 2ULL, 4ULL}) {
		// 64-bit: q < 2^61 (the pre-shift Barrett; same bound as EltwiseFMAMod),
		// which also keeps imf*q < 2^63. 32-bit: imf*q <= 2^32.
		auto ok = [imf](uint64_t q) {
			// (written as a division: imf * q itself can wrap past 2^64)
			return Is32<Word>() ? q <= (1ULL << 32) / imf
				: q < (1ULL << 61) && imf * q < (1ULL << 63);
		};
		for (uint64_t q : Moduli(ok)) {
			for (size_t n : O::EltwiseSizes()) {
				auto a64 = O::Random(n, imf * q), b64 = O::Random(n, imf * q);
				a64[0] = imf * q - 1; b64[0] = imf * q - 1;
				std::vector<Word> want(n);
				for (size_t i = 0; i < n; ++i) want[i] = static_cast<Word>(O::MulMod(a64[i], b64[i], q));
				const auto a = O::As<Word>(a64), b = O::As<Word>(b64);
				CheckBothWays(want, a, [&](Word* r, const Word* x) {
						EltwiseMultMod(r, x, b.data(), n, q, imf); }, "MultMod", q);
			}
		}
	}
}

template <typename Word>
void RunFMAMod() {
	for (uint64_t imf : {1ULL, 2ULL, 4ULL, 8ULL}) {
		// 64-bit: q < 2^61. 32-bit: imf*q <= 2^32.
		auto ok = [imf](uint64_t q) {
			return Is32<Word>() ? q <= (1ULL << 32) / imf : q < (1ULL << 61);
		};
		for (uint64_t q : Moduli(ok)) {
			for (size_t n : O::EltwiseSizes()) {
				auto a64 = O::Random(n, imf * q), c64 = O::Random(n, imf * q);
				const uint64_t s = O::Random(1, imf * q)[0];
				std::vector<Word> want_add(n), want_mul(n);
				for (size_t i = 0; i < n; ++i) {
					const uint64_t m = O::MulMod(a64[i] % q, s % q, q);
					want_mul[i] = static_cast<Word>(m);
					want_add[i] = static_cast<Word>(O::AddMod(m, c64[i] % q, q));
				}
				const auto a = O::As<Word>(a64), c = O::As<Word>(c64);
				CheckBothWays(want_add, a, [&](Word* r, const Word* x) {
						EltwiseFMAMod(r, x, s, c.data(), n, q, imf); }, "FMAMod +arg3", q);
				CheckBothWays(want_mul, a, [&](Word* r, const Word* x) {
						EltwiseFMAMod(r, x, s, static_cast<const Word*>(nullptr), n, q, imf); },
						"FMAMod no arg3", q);
				// result aliasing arg3 (OpenFHE's hexldcrtpoly does this)
				auto c2 = c;
				EltwiseFMAMod(c2.data(), a.data(), s, c2.data(), n, q, imf);
				CHECK_VEC_EQ(want_add, c2, << "FMAMod result==arg3 q=" << q);
			}
		}
	}
}

template <typename Word>
void RunReduceMod() {
	struct Case { uint64_t in; uint64_t out; };  // in == 0 means "modulus"
	const Case cases[] = {{0, 1}, {0, 2}, {2, 1}, {4, 1}, {4, 2}, {2, 2}};
	// 64-bit: q < 2^62. 32-bit: q <= 2^32.
	auto ok = [](uint64_t q) { return Is32<Word>() ? q <= (1ULL << 32) : q < (1ULL << 62); };
	for (uint64_t q : Moduli(ok)) {
		for (const Case& c : cases) {
			if (!c.in && q <= 4) continue;  // imf == q would alias the 2/4 codes
			const uint64_t imf = c.in ? c.in : q;
			if (c.in && (Is32<Word>() ? imf * q > (1ULL << 32) : imf * q >= (1ULL << 63))) continue;
			for (size_t n : O::EltwiseSizes()) {
				// "modulus" means any value of the word
				const auto a64 = c.in ? O::Random(n, imf * q) : O::Random(n, O::AnyWord<Word>());
				const auto a = O::As<Word>(a64);
				std::vector<Word> out(n);
				EltwiseReduceMod(out.data(), a.data(), n, q, imf, c.out);
				for (size_t i = 0; i < n; ++i) {
					// Any representative in [0, out*q) congruent mod q is correct.
					if (c.in == c.out) {
						CHECK_EQ(out[i], a[i], << "copy case i=" << i);
					} else {
						CHECK(out[i] < c.out * q);
						CHECK_EQ(out[i] % q, a[i] % q, << "i=" << i << " q=" << q
								<< " in=" << imf << " out=" << c.out);
					}
				}
				// In place is only defined when the factors differ (upstream asserts
				// input_mod_factor != output_mod_factor once operand == result).
				if (c.in != c.out) {
					auto inplace = a;
					EltwiseReduceMod(inplace.data(), inplace.data(), n, q, imf, c.out);
					for (size_t i = 0; i < n; ++i) {
						CHECK(inplace[i] < c.out * q);
						CHECK_EQ(inplace[i] % q, a[i] % q);
					}
				}
			}
		}
	}
}

template <typename Word>
void RunCmpAdd() {
	for (CMPINT cmp : kCmps) {
		for (size_t n : O::EltwiseSizes()) {
			const uint64_t range = 1000;  // small range so EQ actually hits
			const auto a64 = O::Random(n, range);
			const uint64_t bound = O::Random(1, range)[0];
			const uint64_t diff = 1 + O::Random(1, Is32<Word>() ? (1ULL << 20) : (1ULL << 40))[0];
			std::vector<Word> want(n);
			for (size_t i = 0; i < n; ++i) {
				want[i] = static_cast<Word>(Compare(cmp, a64[i], bound) ? a64[i] + diff : a64[i]);
			}
			CheckBothWays(want, O::As<Word>(a64), [&](Word* r, const Word* x) {
					EltwiseCmpAdd(r, x, n, cmp, bound, diff); }, "CmpAdd",
					static_cast<uint64_t>(cmp));
		}
	}
}

template <typename Word>
void RunCmpSubMod() {
	// 64-bit: q < 2^62. 32-bit: q <= 2^32.
	auto ok = [](uint64_t q) { return q >= 3 && (Is32<Word>() ? q <= (1ULL << 32) : q < (1ULL << 62)); };
	for (uint64_t q : Moduli(ok)) {
		for (CMPINT cmp : kCmps) {
			for (size_t n : O::EltwiseSizes()) {
				// Mix of in-range, just-above-q and full-width values; the bound sits
				// inside the data so every predicate splits the vector.
				const uint64_t span = (4 * q < q || 4 * q > O::AnyWord<Word>()) ? O::AnyWord<Word>() : 4 * q;
				auto a64 = O::Random(n, span);
				for (size_t i = 0; i < n; i += 5) a64[i] = O::Random(1, O::AnyWord<Word>())[0];
				const uint64_t bound = a64[n / 2];
				const uint64_t diff = 1 + O::Random(1, q - 1)[0];
				std::vector<Word> want(n);
				for (size_t i = 0; i < n; ++i) {
					const uint64_t r = a64[i] % q;
					want[i] = static_cast<Word>(Compare(cmp, a64[i], bound) ? O::SubMod(r, diff, q) : r);
				}
				CheckBothWays(want, O::As<Word>(a64), [&](Word* r, const Word* x) {
						EltwiseCmpSubMod(r, x, n, q, cmp, bound, diff); }, "CmpSubMod", q);
			}
		}
	}
}

// The exact call pattern of OpenFHE's NativeVector::SwitchModulus
// (mubintvecnathexl.cpp) when moving to a smaller and to a larger modulus.
// Moduli as OpenFHE uses them: <= 60 bits at NATIVE_SIZE=64, <= 28 at 32.
template <typename Word>
void RunSwitchModulusPattern() {
	const uint64_t om = O::NttPrime(Is32<Word>() ? 24 : 49, 1024);  // old modulus
	const uint64_t nms[] = {O::NttPrime(Is32<Word>() ? 18 : 27, 1024),
		O::NttPrime(Is32<Word>() ? 27 : 55, 1024)};
	for (uint64_t nm : nms) {
		const size_t n = 1024;
		const auto v64 = O::Random(n, om);
		auto got = O::As<Word>(v64);
		if (nm < om) {
			EltwiseCmpSubMod(got.data(), got.data(), n, nm, CMPINT::NLE, om >> 1,
					(om - nm) % nm);
		} else {
			EltwiseCmpAdd(got.data(), got.data(), n, CMPINT::NLE, om >> 1, nm - om);
		}
		for (size_t i = 0; i < n; ++i) {
			// centred lift: values above om/2 represent negatives
			const bool neg = v64[i] > (om >> 1);
			uint64_t want;
			if (nm < om) {
				want = neg ? O::SubMod(v64[i] % nm, (om - nm) % nm, nm) : v64[i] % nm;
			} else {
				want = neg ? v64[i] + (nm - om) : v64[i];
			}
			CHECK_EQ(static_cast<uint64_t>(got[i]), want, << "i=" << i << " nm=" << nm);
		}
	}
}

#ifdef HEXL_RVV_HAS_MULT_ADD_MOD
/// EltwiseMultAddMod (rvv-hexl extension): (a * b + c) mod q, every input in
/// [0, imf * q), every aliasing pattern OpenFHE can produce (result == a,
/// result == b, accumulation result == c).
template <typename Word>
void RunMultAddMod() {
	for (uint64_t imf : {1ULL, 2ULL, 4ULL}) {
		// same bounds as EltwiseMultMod
		auto ok = [imf](uint64_t q) {
			return Is32<Word>() ? q <= (1ULL << 32) / imf
				: q < (1ULL << 61) && imf * q < (1ULL << 63);
		};
		for (uint64_t q : Moduli(ok)) {
			for (size_t n : O::EltwiseSizes()) {
				auto a64 = O::Random(n, imf * q), b64 = O::Random(n, imf * q), c64 = O::Random(n, imf * q);
				a64[0] = imf * q - 1; b64[0] = imf * q - 1; c64[0] = imf * q - 1;  // largest a * b + c
				std::vector<Word> want(n);
				for (size_t i = 0; i < n; ++i)
					want[i] = static_cast<Word>(O::AddMod(O::MulMod(a64[i], b64[i], q), c64[i] % q, q));
				const auto a = O::As<Word>(a64), b = O::As<Word>(b64), c = O::As<Word>(c64);
				// out of place, and result == arg1
				CheckBothWays(want, a, [&](Word* r, const Word* x) {
						EltwiseMultAddMod(r, x, b.data(), c.data(), n, q, imf); }, "MultAddMod", q);
				// result == arg2
				auto b2 = b;
				EltwiseMultAddMod(b2.data(), a.data(), b2.data(), c.data(), n, q, imf);
				CHECK_VEC_EQ(want, b2, << "MultAddMod result==arg2 n=" << n << " q=" << q << " imf=" << imf);
				// result == arg3: the accumulation acc += a * b
				auto c2 = c;
				EltwiseMultAddMod(c2.data(), a.data(), b.data(), c2.data(), n, q, imf);
				CHECK_VEC_EQ(want, c2, << "MultAddMod result==arg3 n=" << n << " q=" << q << " imf=" << imf);
			}
		}
	}
}
#endif

#ifdef HEXL_RVV_HAS_SIGNED_DIGIT_DECOMPOSE
/// Oracle for SignedDigitDecompose, written independently of OpenFHE's shifts:
/// floor division and remainder on the centred value.
std::vector<std::vector<uint64_t>> SignedDigitsOracle(const std::vector<uint64_t>& x,
		uint64_t q, uint64_t g, uint64_t digits) {
	const __int128 base = static_cast<__int128>(1) << g;
	std::vector<std::vector<uint64_t>> out(digits, std::vector<uint64_t>(x.size()));
	for (size_t j = 0; j < x.size(); ++j) {
		__int128 c = x[j] < q / 2 ? static_cast<__int128>(x[j]) : static_cast<__int128>(x[j]) - q;
		for (uint64_t d = 0; d <= digits; ++d) {
			__int128 low = c % base;
			if (low < 0) low += base;                // low digit in [0, B)
			const __int128 r = low >= base / 2 ? low - base : low;  // signed, in [-B/2, B/2)
			c = (c - r) / base;                      // exact
			if (d > 0) out[d - 1][j] = static_cast<uint64_t>(r < 0 ? r + q : r);  // digit 0 dropped
		}
	}
	return out;
}

/// SignedDigitDecompose (rvv-hexl extension, OpenFHE binfhe): every modulus
/// class of the word (e32 and e64 lanes), gadget bases from 2 to 2^20, more
/// digits than the modulus needs (the extra ones are 0 or q - 1), the boundary
/// values around q / 2 and 0, q - 1.
template <typename Word>
void RunSignedDigitDecompose() {
	auto ok = [](uint64_t q) { return Is32<Word>() ? q < (1ULL << 31) : q < (1ULL << 62); };
	for (uint64_t q : Moduli(ok)) {
		const uint64_t bits = 64 - __builtin_clzll(q);
		for (uint64_t g : {1ULL, 2ULL, 5ULL, 7ULL, 10ULL, 16ULL, 20ULL}) {
			if (g >= bits && g > 1) continue;
			const uint64_t full = (bits + g - 1) / g;  // digits of a full decomposition
			for (uint64_t digits : {uint64_t{1}, full > 1 ? full - 1 : 1, full + 1}) {
				for (size_t n : O::EltwiseSizes()) {
					auto x64 = O::Random(n, q);
					const uint64_t edge[] = {0, 1, q / 2 - 1, q / 2, q / 2 + 1, q - 1};
					for (size_t i = 0; i < n && i < 6; ++i) x64[i] = edge[i] % q;
					const auto want = SignedDigitsOracle(x64, q, g, digits);
					const auto x = O::As<Word>(x64);
					std::vector<std::vector<Word>> got(digits, std::vector<Word>(n, static_cast<Word>(0x5A5A5A5A)));
					std::vector<Word*> ptr(digits);
					for (uint64_t d = 0; d < digits; ++d) ptr[d] = got[d].data();
					SignedDigitDecompose(ptr.data(), x.data(), n, q, g, digits);
					for (uint64_t d = 0; d < digits; ++d) {
						for (size_t i = 0; i < n; ++i) {
							CHECK_EQ(static_cast<uint64_t>(got[d][i]), want[d][i],
									<< "SignedDigitDecompose q=" << q << " g=" << g << " digits=" << digits
									<< " n=" << n << " d=" << d << " i=" << i << " x=" << x64[i]);
						}
					}
				}
			}
		}
	}
}
#endif

#ifdef HEXL_RVV_HAS_EXTERNAL_PRODUCT
/// EltwiseExternalProductMod (rvv-hexl extension, OpenFHE binfhe) against OpenFHE's own
/// call sequence on the public functions: per sum one EltwiseMultMod, then one
/// EltwiseMultAddMod per further digit, the monomial by EltwiseMultMod, the terms added
/// by EltwiseAddMod. The four uses of the blind rotation (CGGI with two keys and
/// monomials, DM overwriting, LMKCDEY accumulating, one key with a monomial), 1, 2 and
/// 6 digit polynomials (6 = STD128), every size tail and modulus class of the word, the
/// largest operands in element 0.
template <typename Word>
void RunExternalProduct() {
	struct Use { const char* name; bool key2, mono, accumulate; };
	const Use uses[] = {{"CGGI", true, true, true}, {"DM", false, false, false},
		{"LMKCDEY", false, false, true}, {"one key + monomial", false, true, true}};
	auto ok = [](uint64_t q) { return Is32<Word>() ? q < (1ULL << 32) : q < (1ULL << 61); };
	for (uint64_t q : Moduli(ok)) {
		for (const Use& u : uses) {
			for (uint64_t k : {uint64_t{1}, uint64_t{2}, uint64_t{6}}) {
				for (size_t n : O::EltwiseSizes()) {
					auto poly = [&]() { auto v = O::Random(n, q); v[0] = q - 1; return O::As<Word>(v); };
					std::vector<std::vector<Word>> dg(k), k1(2 * k), k2(2 * k);
					for (auto& v : dg) v = poly();
					for (auto& v : k1) v = poly();
					if (u.key2) for (auto& v : k2) v = poly();
					const std::vector<Word> m1 = poly(), m2 = poly(), a0 = poly(), a1 = poly();
					std::vector<const Word*> pd(k), p1(2 * k), p2(2 * k);
					for (uint64_t i = 0; i < k; ++i) pd[i] = dg[i].data();
					for (uint64_t i = 0; i < 2 * k; ++i) { p1[i] = k1[i].data(); p2[i] = u.key2 ? k2[i].data() : nullptr; }

					// OpenFHE's sequence
					const std::vector<Word>* acc_in[2] = {&a0, &a1};
					std::vector<Word> want[2];
					for (int j = 0; j < 2; ++j) {
						auto sum = [&](const std::vector<const Word*>& key, const Word* mono) {
							std::vector<Word> t(n);
							EltwiseMultMod(t.data(), pd[0], key[j], n, q, 1);
							for (uint64_t i = 1; i < k; ++i) EltwiseMultAddMod(t.data(), pd[i], key[2 * i + j], t.data(), n, q, 1);
							if (mono) EltwiseMultMod(t.data(), t.data(), mono, n, q, 1);
							return t;
						};
						want[j] = sum(p1, u.mono ? m1.data() : nullptr);
						if (u.key2) {
							const auto t2 = sum(p2, u.mono ? m2.data() : nullptr);
							EltwiseAddMod(want[j].data(), want[j].data(), t2.data(), n, q);
						}
						if (u.accumulate) EltwiseAddMod(want[j].data(), want[j].data(), acc_in[j]->data(), n, q);
					}

					std::vector<Word> g0 = a0, g1 = a1;
					EltwiseExternalProductMod(g0.data(), g1.data(), pd.data(), k, p1.data(), u.mono ? m1.data() : nullptr,
							u.key2 ? p2.data() : nullptr, u.mono ? m2.data() : nullptr, n, q, u.accumulate);
					CHECK_VEC_EQ(want[0], g0, << "ExternalProduct " << u.name << " acc0 q=" << q << " k=" << k << " n=" << n
							<< " word=" << 8 * sizeof(Word));
					CHECK_VEC_EQ(want[1], g1, << "ExternalProduct " << u.name << " acc1 q=" << q << " k=" << k << " n=" << n
							<< " word=" << 8 * sizeof(Word));
				}
			}
		}
	}
}
#endif

#ifdef HEXL_RVV_HAS_DOT_PRODUCT
/// EltwiseDotProductMod (rvv-hexl extension) against the call sequences it replaces, on
/// the public functions: SEAL's key switching (digits from the lazy NTT in [0, 4q), one
/// EltwiseMultAddMod per digit and key component into the accumulator), OpenFHE's
/// KeySwitchBV (zero-initialised results, one EltwiseMultAddMod per digit) and a
/// post-multiplied sum (one key of binfhe's CGGI step); M = 1 and 2 results, 1, 3 and 8
/// terms, every size tail and modulus class of the word, the largest operands in
/// element 0. With EltwiseExternalProductMod too: two calls reproduce a CGGI step.
template <typename Word>
void RunDotProduct() {
	struct Use { const char* name; uint64_t imf, results; bool post, accumulate; };
	const Use uses[] = {{"SEAL key switching", 4, 2, false, true}, {"OpenFHE KeySwitchBV", 1, 2, false, false},
		{"post-multiplied", 1, 2, true, true}, {"one result, lazy input", 2, 1, false, false}};
	for (const Use& u : uses) {
		auto ok = [&u](uint64_t q) {
			return Is32<Word>() ? q <= (1ULL << 32) / u.imf : q < (1ULL << 61) && u.imf * q < (1ULL << 63);
		};
		for (uint64_t q : Moduli(ok)) {
			for (uint64_t k : {uint64_t{1}, uint64_t{3}, uint64_t{8}}) {
				for (size_t n : O::EltwiseSizes()) {
					const uint64_t M = u.results;
					auto poly = [&](uint64_t bound) { auto v = O::Random(n, bound); v[0] = bound - 1; return O::As<Word>(v); };
					std::vector<std::vector<Word>> av(k), bv(k * M), out0(M);
					for (auto& v : av) v = poly(u.imf * q);
					for (auto& v : bv) v = poly(q);
					for (auto& v : out0) v = poly(q);
					const std::vector<Word> post = poly(q);
					std::vector<const Word*> pa(k), pb(k * M);
					for (uint64_t i = 0; i < k; ++i) pa[i] = av[i].data();
					for (uint64_t i = 0; i < k * M; ++i) pb[i] = bv[i].data();

					std::vector<std::vector<Word>> want(M);
					for (uint64_t m = 0; m < M; ++m) {
						std::vector<Word> t(n);
						EltwiseMultMod(t.data(), pa[0], pb[m], n, q, u.imf);
						for (uint64_t i = 1; i < k; ++i) EltwiseMultAddMod(t.data(), pa[i], pb[i * M + m], t.data(), n, q, u.imf);
						if (u.post) EltwiseMultMod(t.data(), t.data(), post.data(), n, q, 1);
						if (u.accumulate) EltwiseAddMod(t.data(), t.data(), out0[m].data(), n, q);
						want[m] = t;
					}
					std::vector<std::vector<Word>> got = out0;
					std::vector<Word*> pg(M);
					for (uint64_t m = 0; m < M; ++m) pg[m] = got[m].data();
					EltwiseDotProductMod(pg.data(), M, pa.data(), k, pb.data(), u.post ? post.data() : nullptr, n, q, u.imf,
							u.accumulate);
					for (uint64_t m = 0; m < M; ++m) {
						CHECK_VEC_EQ(want[m], got[m], << "DotProduct " << u.name << " m=" << m << " q=" << q << " k=" << k
								<< " n=" << n << " word=" << 8 * sizeof(Word));
					}
				}
			}
		}
	}
#ifdef HEXL_RVV_HAS_EXTERNAL_PRODUCT
	// a CGGI step as two dot products (one per key) against EltwiseExternalProductMod
	for (uint64_t q : Moduli([](uint64_t q) { return q < (1ULL << 30); })) {
		for (size_t n : O::EltwiseSizes()) {
			const uint64_t k = 6;
			auto poly = [&]() { auto v = O::Random(n, q); v[0] = q - 1; return O::As<Word>(v); };
			std::vector<std::vector<Word>> dg(k), k1(2 * k), k2(2 * k);
			for (auto& v : dg) v = poly();
			for (auto& v : k1) v = poly();
			for (auto& v : k2) v = poly();
			const auto m1 = poly(), m2 = poly(), a0 = poly(), a1 = poly();
			std::vector<const Word*> pd(k), p1(2 * k), p2(2 * k);
			for (uint64_t i = 0; i < k; ++i) pd[i] = dg[i].data();
			for (uint64_t i = 0; i < 2 * k; ++i) { p1[i] = k1[i].data(); p2[i] = k2[i].data(); }
			auto e0 = a0, e1 = a1, d0 = a0, d1 = a1;
			EltwiseExternalProductMod(e0.data(), e1.data(), pd.data(), k, p1.data(), m1.data(), p2.data(), m2.data(), n, q, true);
			Word* pr[2] = {d0.data(), d1.data()};
			EltwiseDotProductMod(pr, 2, pd.data(), k, p1.data(), m1.data(), n, q, 1, true);
			EltwiseDotProductMod(pr, 2, pd.data(), k, p2.data(), m2.data(), n, q, 1, true);
			CHECK_VEC_EQ(e0, d0, << "DotProduct x2 vs ExternalProduct acc0 q=" << q << " n=" << n);
			CHECK_VEC_EQ(e1, d1, << "DotProduct x2 vs ExternalProduct acc1 q=" << q << " n=" << n);
		}
	}
#endif
}
#endif

#ifdef HEXL_RVV_HAS_BASE_CONVERT
/// The constants of a base conversion from first principles: Q_i = prod_{k != i} q_k,
/// q_hat_inv[i] = Q_i^{-1} mod q_i, q_hat_mod_p[i * K + j] = Q_i mod p_j.
struct BaseConvertCase {
	std::vector<uint64_t> q, p, q_hat_inv, q_hat_mod_p;
	BaseConvertCase(std::vector<uint64_t> q_, std::vector<uint64_t> p_) : q(std::move(q_)), p(std::move(p_)) {
		const size_t L = q.size(), K = p.size();
		q_hat_inv.resize(L);
		q_hat_mod_p.resize(L * K);
		for (size_t i = 0; i < L; ++i) {
			uint64_t hat_qi = 1;
			for (size_t k = 0; k < L; ++k)
				if (k != i) hat_qi = O::MulMod(hat_qi, q[k] % q[i], q[i]);
			q_hat_inv[i] = O::InvModPrime(hat_qi, q[i]);
			for (size_t j = 0; j < K; ++j) {
				uint64_t hat_pj = 1;
				for (size_t k = 0; k < L; ++k)
					if (k != i) hat_pj = O::MulMod(hat_pj, q[k] % p[j], p[j]);
				q_hat_mod_p[i * K + j] = hat_pj;
			}
		}
	}
};

/// BaseConvert (rvv-hexl extension, OpenFHE key switching): the defining sum
/// in plain modular arithmetic for every shape, and for L <= 2 (the CRT value
/// X < Q fits in 128 bits) that each output is X + u Q mod p_j with 0 <= u < L,
/// i.e. that this is OpenFHE's approximate conversion. Input limbs larger and
/// smaller than the output limbs (y_i >= p_j and y_i < p_j), up to 2^60.
void RunBaseConvert() {
	using u128 = unsigned __int128;
	const std::pair<size_t, size_t> shapes[] = {{1, 1}, {1, 4}, {2, 3}, {4, 1}, {4, 5}, {3, 8}, {16, 3}};
	const std::pair<size_t, size_t> bits[] = {{36, 46}, {59, 30}, {30, 59}, {46, 59}, {59, 58}};
	for (auto [L, K] : shapes) {
		for (auto [bq, bp] : bits) {
			const BaseConvertCase c(O::Primes(L, bq, true, 1), O::Primes(K, bp, false, 1));
			for (size_t n : O::EltwiseSizes()) {
				std::vector<std::vector<uint64_t>> x(L);
				for (size_t i = 0; i < L; ++i) {
					x[i] = O::Random(n, c.q[i]);
					x[i][0] = c.q[i] - 1;
					if (n > 1) x[i][1] = 0;
				}
				std::vector<std::vector<uint64_t>> got(K, std::vector<uint64_t>(n, 0x5A5A5A5A5A5A5A5AULL));
				std::vector<const uint64_t*> in(L);
				std::vector<uint64_t*> out(K);
				for (size_t i = 0; i < L; ++i) in[i] = x[i].data();
				for (size_t j = 0; j < K; ++j) out[j] = got[j].data();
				BaseConvert(out.data(), in.data(), n, c.q.data(), L, c.p.data(), K, c.q_hat_inv.data(),
						c.q_hat_mod_p.data());
				for (size_t k = 0; k < n; ++k) {
					for (size_t j = 0; j < K; ++j) {
						uint64_t want = 0;
						for (size_t i = 0; i < L; ++i) {
							const uint64_t y = O::MulMod(x[i][k], c.q_hat_inv[i], c.q[i]);
							want = O::AddMod(want, O::MulMod(y % c.p[j], c.q_hat_mod_p[i * K + j], c.p[j]), c.p[j]);
						}
						CHECK_EQ(got[j][k], want, << "BaseConvert L=" << L << " K=" << K << " bits=" << bq << "/" << bp
								<< " n=" << n << " j=" << j << " k=" << k);
					}
					if (L <= 2) {  // X = the CRT value of (x_0, .., x_{L-1}), Q < 2^120
						u128 Q = 1;
						for (size_t i = 0; i < L; ++i) Q *= c.q[i];
						u128 X = 0;
						for (size_t i = 0; i < L; ++i) {
							const uint64_t y = O::MulMod(x[i][k], c.q_hat_inv[i], c.q[i]);
							X = (X + static_cast<u128>(y) * (Q / c.q[i])) % Q;
						}
						for (size_t j = 0; j < K; ++j) {
							bool found = false;
							for (size_t u = 0; u < L && !found; ++u)
								found = got[j][k] == static_cast<uint64_t>((X + u * Q) % c.p[j]);
							CHECK_EQ(found, true, << "BaseConvert is not X + uQ mod p: L=" << L << " K=" << K << " bits="
									<< bq << "/" << bp << " n=" << n << " j=" << j << " k=" << k);
						}
					}
				}
			}
		}
	}
}
#endif

#ifdef HEXL_RVV_HAS_BASE_CONVERT_EXACT
/// BaseConvertExact (rvv-hexl extension, OpenFHE SwitchCRTBasis): OpenFHE's procedure
/// (y_i, the double nu = 0.5 + sum y_i / q_i, alpha = floor(nu), the sum minus alpha Q
/// mod p_j) for every shape, and for L <= 2 (Q < 2^120) that each output is exactly the
/// centred lift of the CRT value X (X, or X - Q when X >= Q/2) mod p_j. The HPS shapes of BFV (3 x 60-bit Q, R one limb
/// larger) and the BaseConvert shapes.
void RunBaseConvertExact() {
	using u128 = unsigned __int128;
	const std::pair<size_t, size_t> shapes[] = {{1, 1}, {2, 3}, {3, 4}, {4, 3}, {4, 5}, {16, 3}};
	const std::pair<size_t, size_t> bits[] = {{59, 59}, {36, 46}, {59, 30}, {30, 59}};
	for (auto [L, K] : shapes) {
		for (auto [bq, bp] : bits) {
			const BaseConvertCase c(O::Primes(L, bq, true, 1), O::Primes(K, bp, false, 1));
			std::vector<double> q_inv(L);
			for (size_t i = 0; i < L; ++i) q_inv[i] = 1.0 / static_cast<double>(c.q[i]);
			std::vector<uint64_t> alpha_q((L + 1) * K);  // a Q mod p_j
			for (size_t j = 0; j < K; ++j) {
				uint64_t Qp = 1;
				for (size_t i = 0; i < L; ++i) Qp = O::MulMod(Qp, c.q[i] % c.p[j], c.p[j]);
				for (size_t a = 0; a <= L; ++a) alpha_q[a * K + j] = O::MulMod(a % c.p[j], Qp, c.p[j]);
			}
			for (size_t n : O::EltwiseSizes()) {
				std::vector<std::vector<uint64_t>> x(L);
				for (size_t i = 0; i < L; ++i) {
					x[i] = O::Random(n, c.q[i]);
					x[i][0] = c.q[i] - 1;
					if (n > 1) x[i][1] = 0;
				}
				std::vector<std::vector<uint64_t>> got(K, std::vector<uint64_t>(n, 0x5A5A5A5A5A5A5A5AULL));
				std::vector<const uint64_t*> in(L);
				std::vector<uint64_t*> out(K);
				for (size_t i = 0; i < L; ++i) in[i] = x[i].data();
				for (size_t j = 0; j < K; ++j) out[j] = got[j].data();
				BaseConvertExact(out.data(), in.data(), n, c.q.data(), L, c.p.data(), K, c.q_hat_inv.data(),
						c.q_hat_mod_p.data(), q_inv.data(), alpha_q.data());
				for (size_t k = 0; k < n; ++k) {
					std::vector<uint64_t> y(L);
					double nu = 0.5;  // OpenFHE's order
					for (size_t i = 0; i < L; ++i) {
						y[i] = O::MulMod(x[i][k], c.q_hat_inv[i], c.q[i]);
						nu += static_cast<double>(y[i]) * q_inv[i];
					}
					const size_t alpha = static_cast<size_t>(nu);
					for (size_t j = 0; j < K; ++j) {
						uint64_t want = 0;
						for (size_t i = 0; i < L; ++i)
							want = O::AddMod(want, O::MulMod(y[i] % c.p[j], c.q_hat_mod_p[i * K + j], c.p[j]), c.p[j]);
						want = O::SubMod(want, alpha_q[alpha * K + j], c.p[j]);
						CHECK_EQ(got[j][k], want, << "BaseConvertExact L=" << L << " K=" << K << " bits=" << bq << "/"
								<< bp << " n=" << n << " j=" << j << " k=" << k);
					}
					if (L <= 2) {  // exact: the centred lift of X, in [-Q/2, Q/2), mod p_j
						u128 Q = 1;
						for (size_t i = 0; i < L; ++i) Q *= c.q[i];
						u128 X = 0;
						for (size_t i = 0; i < L; ++i) X = (X + static_cast<u128>(y[i]) * (Q / c.q[i])) % Q;
						const bool neg = 2 * X >= Q;  // X - Q
						for (size_t j = 0; j < K; ++j) {
							const uint64_t pj = c.p[j], xm = static_cast<uint64_t>(X % pj), qm = static_cast<uint64_t>(Q % pj);
							CHECK_EQ(got[j][k], neg ? O::SubMod(xm, qm, pj) : xm, << "BaseConvertExact is not the centred X mod p: L="
									<< L << " K=" << K << " bits=" << bq << "/" << bp << " n=" << n << " j=" << j << " k=" << k);
						}
					}
				}
			}
		}
	}
}
#endif

}  // namespace

TEST(Eltwise_AddMod) { RunAddSub<uint64_t>(true); }
TEST(Eltwise_SubMod) { RunAddSub<uint64_t>(false); }
TEST(Eltwise_MultMod) { RunMultMod<uint64_t>(); }
TEST(Eltwise_FMAMod) { RunFMAMod<uint64_t>(); }
TEST(Eltwise_ReduceMod) { RunReduceMod<uint64_t>(); }
TEST(Eltwise_CmpAdd) { RunCmpAdd<uint64_t>(); }
TEST(Eltwise_CmpSubMod) { RunCmpSubMod<uint64_t>(); }
TEST(Eltwise_OpenFHESwitchModulusPattern) { RunSwitchModulusPattern<uint64_t>(); }
#ifdef HEXL_RVV_HAS_MULT_ADD_MOD
TEST(Eltwise_MultAddMod) { RunMultAddMod<uint64_t>(); }
#endif
#ifdef HEXL_RVV_HAS_SIGNED_DIGIT_DECOMPOSE
TEST(Eltwise_SignedDigitDecompose) { RunSignedDigitDecompose<uint64_t>(); }
#endif
#ifdef HEXL_RVV_HAS_BASE_CONVERT
TEST(Eltwise_BaseConvert) { RunBaseConvert(); }
#endif
#ifdef HEXL_RVV_HAS_BASE_CONVERT_EXACT
TEST(Eltwise_BaseConvertExact) { RunBaseConvertExact(); }
#endif
#ifdef HEXL_RVV_HAS_EXTERNAL_PRODUCT
TEST(Eltwise_ExternalProduct) { RunExternalProduct<uint64_t>(); }
#endif
#ifdef HEXL_RVV_HAS_DOT_PRODUCT
TEST(Eltwise_DotProduct) { RunDotProduct<uint64_t>(); }
#endif

#ifdef HEXL_RVV_HAS_32BIT_API
TEST(Eltwise_AddMod_32) { RunAddSub<uint32_t>(true); }
TEST(Eltwise_SubMod_32) { RunAddSub<uint32_t>(false); }
TEST(Eltwise_MultMod_32) { RunMultMod<uint32_t>(); }
TEST(Eltwise_FMAMod_32) { RunFMAMod<uint32_t>(); }
TEST(Eltwise_ReduceMod_32) { RunReduceMod<uint32_t>(); }
TEST(Eltwise_CmpAdd_32) { RunCmpAdd<uint32_t>(); }
TEST(Eltwise_CmpSubMod_32) { RunCmpSubMod<uint32_t>(); }
TEST(Eltwise_OpenFHESwitchModulusPattern_32) { RunSwitchModulusPattern<uint32_t>(); }
#ifdef HEXL_RVV_HAS_MULT_ADD_MOD
TEST(Eltwise_MultAddMod_32) { RunMultAddMod<uint32_t>(); }
#endif
#ifdef HEXL_RVV_HAS_SIGNED_DIGIT_DECOMPOSE
TEST(Eltwise_SignedDigitDecompose_32) { RunSignedDigitDecompose<uint32_t>(); }
#endif
#ifdef HEXL_RVV_HAS_EXTERNAL_PRODUCT
TEST(Eltwise_ExternalProduct_32) { RunExternalProduct<uint32_t>(); }
#endif
#ifdef HEXL_RVV_HAS_DOT_PRODUCT
TEST(Eltwise_DotProduct_32) { RunDotProduct<uint32_t>(); }
#endif
#endif
