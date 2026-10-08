// direction_a_lwe_openfhe_bench.cpp — OpenFHE port of the Direction-A central FHE path,
// instrumented to measure OpenMP MULTI-CORE SCALING (a CPU proxy for what a CAR-V vector/NTT
// cluster or multicore on the S32N79 would buy).
//
// WHY: SEAL's core NTT/relin is single-threaded and scalar on ARM (verified: no NEON, no threads),
// so the ~1.65 s central path is a single-core number. OpenFHE parallelizes its primitives with
// OpenMP (over RNS limbs / NTT / key-switch towers) — the SAME latent RNS/limb parallelism SEAL
// leaves on the table. This bench runs the two dominant central steps — EvalTrace promote
// (log2 N automorphisms + adds, depth 0) and the degree-5 decision (27 ciphertext mults, depth 3) —
// under a sweep of OMP thread counts and reports the speedup vs single-thread.
//
// NOTE: this is a FIDELITY-FOR-PARALLELISM port. It reproduces the FHE-heavy operations (promote +
// decision) that are ~98% of the central cost; it does NOT re-implement the seeded-LWE wire / CRT
// plumbing (<2% of cost, irrelevant to the threading question). Numbers are comparable in SHAPE to
// direction_a_lwe_bench (SEAL), not byte-identical.
//
// Build: see benches/build_openfhe.sh for the library; compile line in direction_a_lwe_openfhe_results.md.

#include "openfhe.h"
#include <omp.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

using namespace lbcrypto;
using namespace std;
using Clk = chrono::high_resolution_clock;
static double ms_since(Clk::time_point t){ return chrono::duration<double,milli>(Clk::now()-t).count(); }

// EvalTrace: kill non-constant coeffs via log2(N) automorphisms X->X^(k+1) + adds (depth 0).
static Ciphertext<DCRTPoly> eval_trace(Ciphertext<DCRTPoly> ct, size_t N,
		CryptoContext<DCRTPoly>& cc, const std::map<usint,EvalKey<DCRTPoly>>& autoKeys){
	Ciphertext<DCRTPoly> acc = ct;
	for(size_t k=N;k>=2;k/=2){
		auto t = cc->EvalAutomorphism(acc, (usint)(k+1), autoKeys);
		acc = cc->EvalAdd(acc, t);
	}
	return acc;
}

// polynomial multiply of two ciphertext-vectors (convolution), each EvalMult auto-relinearized.
static vector<Ciphertext<DCRTPoly>> poly_mul(const vector<Ciphertext<DCRTPoly>>&a,
		const vector<Ciphertext<DCRTPoly>>&b, CryptoContext<DCRTPoly>& cc){
	vector<Ciphertext<DCRTPoly>> out(a.size()+b.size()-1);
	vector<bool> init(out.size(),false);
	for(size_t i=0;i<a.size();++i) for(size_t j=0;j<b.size();++j){
		auto t = cc->EvalMult(a[i], b[j]);          // includes relinearization
		if(!init[i+j]){ out[i+j]=t; init[i+j]=true; } else out[i+j]=cc->EvalAdd(out[i+j],t);
	}
	return out;
}

int main(int argc, char**argv){
	size_t N=16384; int rounds=3; const int KFB=5; int depth=3; int dcrtbits=0;
	vector<int> threads = {1,2,4,6,8,12,18};
	for(int i=1;i<argc;++i){ string a=argv[i];
		if(a=="--N"&&i+1<argc) N=(size_t)atol(argv[++i]);
		else if(a=="--rounds"&&i+1<argc) rounds=atoi(argv[++i]);
		else if(a=="--depth"&&i+1<argc) depth=atoi(argv[++i]); // more depth => more RNS towers => wider limb-parallelism
		else if(a=="--dcrtbits"&&i+1<argc) dcrtbits=atoi(argv[++i]); // RNS prime bit-size (e.g. 48 to match BFVDefault)
	}

	cout<<"Direction A — OpenFHE BFV central path, OpenMP scaling probe\n";
	cout<<"N="<<N<<", KFB="<<KFB<<", rounds/measure="<<rounds<<", max HW threads="<<omp_get_max_threads()<<"\n\n";

	// ---- BFV context: depth 3 (degree-5 decision), batching plaintext modulus 65537 (=2^16+1, NTT-friendly) ----
	CCParams<CryptoContextBFVRNS> parameters;
	parameters.SetPlaintextModulus(65537);
	parameters.SetMultiplicativeDepth(depth);
	if(dcrtbits>0) parameters.SetScalingModSize(dcrtbits); // force RNS prime bit-size (match BFVDefault ~48-bit)
	parameters.SetRingDim(N);
	parameters.SetSecurityLevel(HEStd_128_classic);
	CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
	cc->Enable(PKE); cc->Enable(KEYSWITCH); cc->Enable(LEVELEDSHE); cc->Enable(ADVANCEDSHE);
	{
		auto ep = cc->GetCryptoParameters()->GetElementParams()->GetParams();
		size_t totbits=0; for(auto&p:ep) totbits += p->GetModulus().GetMSB();
		cout<<"ring dim="<<cc->GetRingDimension()<<", # RNS limbs (towers)="<<ep.size()
			<<", prime0 bits="<<(ep.empty()?0:ep[0]->GetModulus().GetMSB())
			<<", total modulus bits="<<totbits<<"\n\n";
	}

	auto keys = cc->KeyGen();
	cc->EvalMultKeyGen(keys.secretKey);
	// automorphism keys for the EvalTrace indices {N+1, N/2+1, ..., 3}
	vector<usint> idx; for(size_t k=N;k>=2;k/=2) idx.push_back((usint)(k+1));
	auto autoKeyMap = cc->EvalAutomorphismKeyGen(keys.secretKey, idx);

	// ---- encrypt 10 packed values (5 feedback + 5 MAC slopes), each a degree-1 pair for the tree ----
	auto mk = [&](int64_t v){ vector<int64_t> vec(cc->GetRingDimension()/2, v);
		return cc->Encrypt(keys.publicKey, cc->MakePackedPlaintext(vec)); };
	vector<Ciphertext<DCRTPoly>> m_ct(KFB), s_ct(KFB);
	for(int i=0;i<KFB;++i){ m_ct[i]=mk(1234+i); s_ct[i]=mk(7+i); }

	// ---- correctness sanity: promote one, decrypt, confirm it round-trips (constant slot) ----
	{ auto p = eval_trace(m_ct[0], N, cc, *autoKeyMap);
		Plaintext out; cc->Decrypt(keys.secretKey, p, &out);
		cout<<"promote round-trip (slot0 after trace, expect N*m mod t): "<<out->GetPackedValue()[0]<<"\n\n"; }

	cout<<fixed<<setprecision(1);
	cout<<"  threads |   promote(10) ms |  decision ms |   total ms |  speedup\n";
	cout<<"  --------+------------------+--------------+------------+---------\n";
	double base_total=0;
	const int HWMAX = omp_get_num_procs();   // real hardware count (omp_get_max_threads mutates with set)
	for(int T: threads){
		if(T>HWMAX) continue;
		omp_set_num_threads(T);
		double tp=0, td=0;
		for(int r=0; r<rounds; ++r){
			// promote all 10
			auto t0=Clk::now();
			vector<Ciphertext<DCRTPoly>> pm(KFB), ps(KFB);
			for(int i=0;i<KFB;++i){ pm[i]=eval_trace(m_ct[i],N,cc,*autoKeyMap); ps[i]=eval_trace(s_ct[i],N,cc,*autoKeyMap); }
			tp += ms_since(t0);
			// degree-5 decision: Y = prod_{i=0..4} (pm[i] + ps[i] X)  via the same tree as SEAL
			auto t1=Clk::now();
			vector<vector<Ciphertext<DCRTPoly>>> y(KFB);
			for(int i=0;i<KFB;++i){ y[i]={pm[i], ps[i]}; }
			auto T12=poly_mul(y[0],y[1],cc); auto T34=poly_mul(y[2],y[3],cc);
			auto T1234=poly_mul(T12,T34,cc); auto Y=poly_mul(T1234,y[4],cc);
			td += ms_since(t1);
			(void)Y;
		}
		tp/=rounds; td/=rounds;
		double tot=tp+td;
		if(T==1) base_total=tot;
		cout<<"  "<<setw(7)<<T<<" | "<<setw(16)<<tp<<" | "<<setw(12)<<td<<" | "
			<<setw(10)<<tot<<" | "<<setw(6)<<(base_total>0?base_total/tot:1.0)<<"x\n";
	}
	cout<<"\n  (promote = 10x EvalTrace automorphism sums; decision = 27 ciphertext mults, depth 3)\n";
	cout<<"  Interpretation: speedup at T>1 is the multi-core headroom SEAL's single-thread path leaves\n";
	cout<<"  unused — the CPU analogue of the CAR-V limb/NTT parallelism on the S32N79.\n";
	return 0;
}
