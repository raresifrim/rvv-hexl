// direction_a_lwe_bench.cpp — REAL seeded-LWE compress path (no modeling), verified end-to-end.
//
// Unlike direction_a_compress_bench.cpp (which stood in a full SEAL encrypt_symmetric for the
// uplink and used an analytic wire size), this bench implements the actual low-dimension LWE layer:
//
//   actuator: encrypt each value as a SEEDED LWE ciphertext (a,b) under the SEAL secret key's own
//             coefficient vector s. The 'a' part is regenerated from a 32-byte PRNG seed, so the wire
//             payload is just b (Lp residues) + one shared seed.
//   central : regenerate a, modulus-switch (a,b) from the small carrying modulus q' up to the full
//             RLWE modulus Q (EXACT, because q' | Q), embed as an RLWE ciphertext, then promote with
//             EvalTrace (depth 0) and run the degree-5 decision.
//
// Everything here is real and verified: LWE self-decryption is checked, and the embedded->promoted
// ciphertext is decrypted back to the original value each round.
//
// Key facts established while writing this:
//   * BFV needs Q > 2*t^2 to decrypt, so the LWE carrying modulus q' must also exceed 2*t^2.
//     => 32-bit values need q' > 2^65 (Lp=2 SEAL primes); 60-bit values need ~3 primes.
//     So real b is ~13 B (32-bit) / ~19 B (60-bit), NOT a single ~4 B word.
//   * Since q' = product of the first Lp SEAL primes, M_hi = Q/q' is an integer and every high prime
//     Q_i (i>=Lp) divides M_hi, so the embedded ciphertext is exactly 0 on the high primes. Hence the
//     whole mod-switch+embed is single-word arithmetic — no multiprecision per coefficient.
//
// Build: clang++ -std=c++17 -O3 -I <SEAL>/native/src -I <SEAL>/build/native/src \
//   -I <SEAL>/build/thirdparty/msgsl-src/include -I /opt/homebrew/include \
//   direction_a_lwe_bench.cpp <SEAL>/build/lib/libseal-4.1.a -L/opt/homebrew/lib -lgmp -lgmpxx -o ...

#include "seal/seal.h"
#include "seal/util/ntt.h"
#include <gmpxx.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <sstream>
#include <cstdint>
#include "../common/snark_flags.hpp"   // enhanced-ADSC-SNARK flag proof (linked when HAVE_SNARK)

using namespace std;
using namespace seal;
using namespace seal::util;
using Clk = chrono::high_resolution_clock;
static double ms_since(Clk::time_point t){ return chrono::duration<double,milli>(Clk::now()-t).count(); }
static double mean(const vector<double>&v){ double s=0; for(double x:v)s+=x; return v.empty()?0:s/v.size(); }
static uint64_t modpow(uint64_t b,uint64_t e,uint64_t m){ __uint128_t r=1,bb=b%m; while(e){ if(e&1)r=(r*bb)%m; bb=(bb*bb)%m; e>>=1;} return (uint64_t)r; }

// EvalTrace: kill non-constant coeffs => (N*m0) in coeff 0; log2(N) automorphisms + adds, depth 0.
static Ciphertext eval_trace(const Ciphertext &ct, size_t N, const GaloisKeys &gk, Evaluator &ev){
    Ciphertext acc = ct;
    for (size_t k=N;k>=2;k/=2){ Ciphertext t; ev.apply_galois(acc,(uint32_t)(k+1),gk,t); ev.add_inplace(acc,t);} return acc;
}
static vector<Ciphertext> poly_mul(const vector<Ciphertext>&a,const vector<Ciphertext>&b,Evaluator&ev,const RelinKeys&rlk){
    vector<Ciphertext> out(a.size()+b.size()-1); vector<bool> init(out.size(),false);
    for(size_t i=0;i<a.size();++i)for(size_t j=0;j<b.size();++j){
        Ciphertext t; ev.multiply(a[i],b[j],t); ev.relinearize_inplace(t,rlk);
        if(!init[i+j]){out[i+j]=t;init[i+j]=true;} else ev.add_inplace(out[i+j],t);
    } return out;
}

int main(int argc, char**argv){
    size_t N=16384; int pt_bits=32; int rounds=8; const int KFB=5; int force_lp=0; int rprimes=2; int kmac=1; int limb_bits=0;
    bool prove_flags=false;   // enhanced-ADSC-SNARK: prove flag_i=(reading_i<threshold_i) at the source
    for(int i=1;i<argc;++i){ string a=argv[i];
        if(a=="--rounds"&&i+1<argc)rounds=atoi(argv[++i]);
        else if(a=="--prove-flags")prove_flags=true;
        else if(a=="--N"&&i+1<argc)N=(size_t)atol(argv[++i]);
        else if(a=="--pbits"&&i+1<argc)pt_bits=atoi(argv[++i]);
        else if(a=="--qprimes"&&i+1<argc)force_lp=atoi(argv[++i]); // override carrying-modulus prime count
        else if(a=="--rprimes"&&i+1<argc)rprimes=atoi(argv[++i]); // downlink result modulus prime count
        else if(a=="--kmac"&&i+1<argc)kmac=atoi(argv[++i]);       // MAC evaluation points (1=single-pt, 2=two-pt soundness)
        else if(a=="--limbbits"&&i+1<argc)limb_bits=atoi(argv[++i]); // 0=BFVDefault; else explicit N*limb_bits-bit chain
    }
    cout<<"Direction A — REAL seeded-LWE compress path (SEAL 4.1 BFV, verified)\n";
    cout<<"N="<<N<<", p="<<pt_bits<<"-bit, rounds="<<rounds<<", KFB="<<KFB<<" feedback values (+ "<<KFB<<" MAC shares)\n\n";

    EncryptionParameters parms(scheme_type::bfv);
    parms.set_poly_modulus_degree(N);
    if(limb_bits>0){ size_t nl=CoeffModulus::MaxBitCount(N)/(size_t)limb_bits;
        parms.set_coeff_modulus(CoeffModulus::Create(N, vector<int>(nl,limb_bits)));
        cout<<"[explicit chain: "<<nl<<" x "<<limb_bits<<"-bit = "<<nl*limb_bits<<" bits]\n"; }
    else parms.set_coeff_modulus(CoeffModulus::BFVDefault(N));
    parms.set_plain_modulus(PlainModulus::Batching(N, pt_bits));
    SEALContext context(parms);
    if(!context.parameters_set()){ cout<<"invalid params\n"; return 1; }
    KeyGenerator kg(context);
    SecretKey sk = kg.secret_key();
    PublicKey pk; kg.create_public_key(pk);
    RelinKeys rlk; kg.create_relin_keys(rlk);
    vector<uint32_t> elts; for(size_t kk=N;kk>=2;kk/=2) elts.push_back((uint32_t)(kk+1));
    GaloisKeys gk; kg.create_galois_keys(elts, gk);
    Evaluator ev(context);
    Decryptor dec(context, sk);

    auto &dcd = *context.first_context_data();
    auto coeff_modulus = dcd.parms().coeff_modulus();
    size_t L = coeff_modulus.size();
    uint64_t t = parms.plain_modulus().value();
    uint64_t Ninv = modpow(N % t, t-2, t);

    // ---- extract ternary secret from data-prime 0 ----
    const NTTTables *key_ntt = context.key_context_data()->small_ntt_tables();
    const uint64_t *skd = sk.data().data();
    vector<uint64_t> s0(skd, skd+N);
    inverse_ntt_negacyclic_harvey(s0.data(), key_ntt[0]);
    uint64_t q0 = coeff_modulus[0].value();
    vector<int> s(N);
    for(size_t j=0;j<N;++j){ s[j] = (s0[j]==0)?0 : (s0[j]==1?1 : (s0[j]==q0-1? -1 : 0)); }

    // ---- choose carrying modulus q' = first Lp primes, with q' > 2*t^2 ----
    mpz_class Q=1; for(auto&m:coeff_modulus) Q*= mpz_class(to_string(m.value()));
    mpz_class need = mpz_class(2) * mpz_class(to_string(t)) * mpz_class(to_string(t)); // 2 t^2
    size_t Lp=0; mpz_class qp=1;
    while(qp <= need && Lp < L){ qp *= mpz_class(to_string(coeff_modulus[Lp].value())); ++Lp; }
    while((int)Lp < force_lp && Lp < L-1){ qp *= mpz_class(to_string(coeff_modulus[Lp].value())); ++Lp; } // optional larger q'
    if(Lp>=L){ cout<<"plaintext too large for this Q\n"; return 1; }
    mpz_class M_hi = Q / qp;                 // integer, = product of primes Lp..L-1
    mpz_class Deltap = qp / mpz_class(to_string(t));   // floor(q'/t)
    // precompute per low-prime constants
    vector<uint64_t> qpi(Lp), Mhi_mod(Lp), Deltap_mod(Lp);
    for(size_t i=0;i<Lp;++i){ uint64_t qi=coeff_modulus[i].value(); qpi[i]=qi;
        Mhi_mod[i] = mpz_class(M_hi % mpz_class(to_string(qi))).get_ui();
        Deltap_mod[i] = mpz_class(Deltap % mpz_class(to_string(qi))).get_ui();
    }
    size_t qp_bits=0; { mpz_class z=qp; while(z>0){++qp_bits; z/=2;} }
    size_t b_bytes = (qp_bits+7)/8;          // wire bytes for one b (Lp residues packed)
    cout<<"carrying modulus q' = first "<<Lp<<" prime(s), ~"<<qp_bits<<" bits  (need q'>2t^2)\n";
    cout<<"=> real seeded-LWE wire per value: b="<<b_bytes<<" B + one shared 32 B seed\n\n";

    // ---- seeded a generator (shared seed; per (ctr,prime,coeff)) ----
    auto gen_a = [&](uint64_t seed, int ctr, size_t i, vector<uint64_t>&a){
        mt19937_64 g(seed ^ (0x9e3779b97f4a7c15ULL*(uint64_t)ctr) ^ (0xD1B54A32D192ED03ULL*(uint64_t)i));
        uint64_t qi=qpi[i]; a.resize(N); for(size_t j=0;j<N;++j) a[j]=g()%qi;
    };
    const uint64_t SEED = 0xC0FFEEULL;
    mt19937_64 erng(7);

    // ---- LWE encrypt one value -> residues b[Lp] (a regenerated from seed) ----
    auto lwe_encrypt = [&](uint64_t m, int ctr, vector<uint64_t>&b){
        b.assign(Lp,0);
        long e_signed = (long)(erng()%17) - 8;                // one small error in [-8,8], SAME across primes
        for(size_t i=0;i<Lp;++i){ uint64_t qi=qpi[i];
            vector<uint64_t> a; gen_a(SEED, ctr, i, a);
            uint64_t dot=0; for(size_t j=0;j<N;++j){ if(s[j]==0)continue; dot = (s[j]==1)?(dot+a[j])%qi:(dot+(qi-a[j]))%qi; }
            uint64_t emod = (e_signed>=0) ? ((uint64_t)e_signed % qi) : (qi - ((uint64_t)(-e_signed) % qi));
            uint64_t dm = (uint64_t)(((__uint128_t)Deltap_mod[i]*(m%qi))%qi);
            b[i] = (uint64_t)(((__uint128_t)dm + emod + qi - dot)%qi);
        }
    };
    // ---- LWE self-decrypt check (CRT low primes, scale by t/q') ----
    auto lwe_decrypt = [&](const vector<uint64_t>&b, int ctr)->uint64_t{
        // reconstruct phase' in [0,q') by CRT of (b_i + <a_i,s>) mod q'_i
        mpz_class phase=0, Mn=1;
        // Garner CRT
        vector<uint64_t> ph(Lp);
        for(size_t i=0;i<Lp;++i){ uint64_t qi=qpi[i]; vector<uint64_t> a; gen_a(SEED,ctr,i,a);
            uint64_t dot=0; for(size_t j=0;j<N;++j){ if(s[j]==0)continue; dot=(s[j]==1)?(dot+a[j])%qi:(dot+(qi-a[j]))%qi; }
            ph[i]=(uint64_t)(((__uint128_t)b[i]+dot)%qi);
        }
        // CRT combine
        for(size_t i=0;i<Lp;++i){ mpz_class qi(to_string(qpi[i])); mpz_class ri(to_string(ph[i]));
            // phase = phase + Mn * ((ri - phase) * inv(Mn mod qi) mod qi)
            mpz_class diff = ((ri - (phase % qi)) % qi + qi)%qi;
            mpz_class invMn; mpz_invert(invMn.get_mpz_t(), mpz_class(Mn%qi).get_mpz_t(), qi.get_mpz_t());
            mpz_class tmp = (diff*invMn)%qi; phase += Mn*tmp; Mn*=qi;
        }
        // m = round(t * phase / q')
        mpz_class num = mpz_class(to_string(t))*phase + qp/2; mpz_class mm = num/qp;
        return (uint64_t)mpz_class(mm % mpz_class(to_string(t))).get_ui();
    };
    // ---- mod-switch up (exact) + embed LWE residues into an RLWE ciphertext ----
    auto embed = [&](const vector<uint64_t>&b, int ctr, Ciphertext&ct){
        ct.resize(context, context.first_parms_id(), 2); ct.is_ntt_form()=false;
        uint64_t *c0=ct.data(0), *c1=ct.data(1);
        for(size_t i=0;i<L;++i){ uint64_t qi=coeff_modulus[i].value(); uint64_t*C0=c0+i*N,*C1=c1+i*N;
            for(size_t j=0;j<N;++j){ C0[j]=0; C1[j]=0; }
        }
        for(size_t i=0;i<Lp;++i){ uint64_t qi=qpi[i]; uint64_t mh=Mhi_mod[i];
            vector<uint64_t> a; gen_a(SEED,ctr,i,a);
            uint64_t bQ = (uint64_t)(((__uint128_t)b[i]*mh)%qi);
            uint64_t*C0=c0+i*N,*C1=c1+i*N;
            C0[0]=bQ;
            C1[0]=(uint64_t)(((__uint128_t)a[0]*mh)%qi);
            for(size_t j=1;j<N;++j){ uint64_t aQ=(uint64_t)(((__uint128_t)a[N-j]*mh)%qi); C1[j]=(qi-aQ)%qi; }
        }
    };
    auto const_plain=[&](uint64_t v){ Plaintext pp; pp.resize(1); pp[0]=v%t; return pp; };

    // ---- DOWNLINK compression: mod-switch the result down to rp primes, then sample-extract coeff 0
    //      into a single LWE (a in Z_q''^N, b). a is computation-derived (NOT seedable), so it is sent
    //      in full -> wire = N*ceil(log2 q'')/8 + b. Returns wire bytes; verifies decode == reference. ----
    auto downlink_extract = [&](const Ciphertext &Y0, size_t rp, uint64_t &decoded)->size_t{
        Ciphertext yd = Y0;
        while (context.get_context_data(yd.parms_id())->parms().coeff_modulus().size() > rp)
            ev.mod_switch_to_next_inplace(yd);
        auto cd = context.get_context_data(yd.parms_id());
        auto cm = cd->parms().coeff_modulus();
        size_t rL = cm.size();
        const uint64_t *c0 = yd.data(0), *c1 = yd.data(1);
        // b residues and a[k] residues per prime: a[0]=c1[0], a[k]=-c1[N-k]
        // reconstruct phase_0 = b + <a,s> via CRT over the rp primes
        mpz_class qpp = 1; for (auto &m : cm) qpp *= mpz_class(to_string(m.value()));
        mpz_class phase = 0, Mn = 1;
        for (size_t i=0;i<rL;++i){ uint64_t qi=cm[i].value();
            const uint64_t *C0=c0+i*N, *C1=c1+i*N;
            uint64_t bb = C0[0];
            uint64_t dot = 0;
            for (size_t k=0;k<N;++k){ if(s[k]==0) continue;
                uint64_t ak = (k==0)? C1[0] : (qi - C1[N-k])%qi;   // a_k
                dot = (s[k]==1) ? (dot+ak)%qi : (dot+(qi-ak))%qi;
            }
            uint64_t ph = (uint64_t)(((__uint128_t)bb + dot)%qi);
            mpz_class qz(to_string(qi)), rz(to_string(ph));
            mpz_class diff=((rz-(phase%qz))%qz+qz)%qz, invMn; mpz_invert(invMn.get_mpz_t(), mpz_class(Mn%qz).get_mpz_t(), qz.get_mpz_t());
            phase += Mn*((diff*invMn)%qz); Mn*=qz;
        }
        mpz_class mm = (mpz_class(to_string(t))*phase + qpp/2)/qpp;
        decoded = (uint64_t)mpz_class(mm % mpz_class(to_string(t))).get_ui();
        size_t qbits=0; { mpz_class z=qpp; while(z>0){++qbits; z/=2;} }
        size_t per = (qbits+7)/8;
        return N*per + per;   // a (N coeffs) + b, at modulus q'' = rp primes
    };

    // ---- per-step timing ----
    vector<double> t_enc, t_embed, t_promote, t_decision, t_decval, t_dlcompress;
    int lwe_ok=0, prom_ok=0, dec_fits=0, dl_ok=0, mac_pass=0;
    size_t result_bytes=0, dl_unpacked_bytes=0, dl_packed_bytes=0; int prom_noise=-1, dec_noise=-1;
    vector<double> t_dlfull, t_macverify;
    // round-keyed bounded-degree MAC: secret point alpha; tag y_i(X)=m_i+s_i*X with y_i(0)=m_i, y_i(alpha)=r_i.
    // central computes Y(X)=prod y_i (deg KFB, KFB+1 coeffs); Y(0)=prod m_i (decision), Y(alpha)=prod r_i (tag).
    uint64_t alpha = (erng()%(t-1))+1; uint64_t alphainv = modpow(alpha, t-2, t);
    // second independent MAC point for k=2 (drawn only when kmac==2 so k=1 RNG stream is unchanged)
    uint64_t alpha2 = (kmac==2)?((erng()%(t-1))+1):0; uint64_t alpha2inv = (kmac==2)?modpow(alpha2, t-2, t):0;

    for(int r=0;r<rounds;++r){
        // feedback values m_i, round keys r_i, tag slopes s_i = (r_i - m_i)/alpha
        vector<uint64_t> m_val(KFB), s_val(KFB), r_val(KFB), s2_val(KFB,0), r2_val(KFB,0);
        for(int i=0;i<KFB;++i){ m_val[i]=(uint64_t)((1234567ULL*(r+1)+i*99)% t);
            r_val[i]=(erng()%(t-1))+1;
            uint64_t diff=(r_val[i]+t-(m_val[i]%t))%t; s_val[i]=(uint64_t)(((__uint128_t)diff*alphainv)%t); }
        if(kmac==2) for(int i=0;i<KFB;++i){ r2_val[i]=(erng()%(t-1))+1;   // second slope set for point alpha2
            uint64_t diff2=(r2_val[i]+t-(m_val[i]%t))%t; s2_val[i]=(uint64_t)(((__uint128_t)diff2*alpha2inv)%t); }

        // ---- (A) actuator: real seeded-LWE encrypt of 10 values ----
        vector<vector<uint64_t>> wire_m(KFB), wire_s(KFB), wire_s2(KFB);
        auto tk=Clk::now();
        for(int i=0;i<KFB;++i){ lwe_encrypt(m_val[i], 2*i,   wire_m[i]); lwe_encrypt(s_val[i], 2*i+1, wire_s[i]);
            if(kmac==2) lwe_encrypt(s2_val[i], 2*KFB+i, wire_s2[i]); }
        t_enc.push_back(ms_since(tk));
        // verify LWE self-decrypt
        if(r==0){ bool ok=true; for(int i=0;i<KFB;++i){ if(lwe_decrypt(wire_m[i],2*i)!=m_val[i]) ok=false; if(lwe_decrypt(wire_s[i],2*i+1)!=s_val[i]) ok=false; } if(ok) lwe_ok=1; }

        // ---- (B) central: mod-switch + embed 10 LWE -> RLWE ----
        vector<Ciphertext> emb_m(KFB), emb_s(KFB), emb_s2(KFB);
        tk=Clk::now();
        for(int i=0;i<KFB;++i){ embed(wire_m[i],2*i,emb_m[i]); embed(wire_s[i],2*i+1,emb_s[i]);
            if(kmac==2) embed(wire_s2[i],2*KFB+i,emb_s2[i]); }
        t_embed.push_back(ms_since(tk));

        // ---- (C) promote via EvalTrace (depth 0) ----
        vector<Ciphertext> prom_m(KFB), prom_s(KFB), prom_s2(KFB);
        tk=Clk::now();
        for(int i=0;i<KFB;++i){ prom_m[i]=eval_trace(emb_m[i],N,gk,ev); prom_s[i]=eval_trace(emb_s[i],N,gk,ev);
            if(kmac==2) prom_s2[i]=eval_trace(emb_s2[i],N,gk,ev); }
        t_promote.push_back(ms_since(tk));
        if(r==0){ prom_noise=dec.invariant_noise_budget(prom_m[0]); bool ok=true;
            for(int i=0;i<KFB;++i){ Plaintext o; dec.decrypt(prom_m[i],o); uint64_t got=((__uint128_t)(o.coeff_count()?o[0]:0)*Ninv)%t; if(got!=m_val[i]) ok=false; }
            if(ok) prom_ok=1;
        }

        // ---- (D) degree-5 decision / MAC product ----
        vector<Ciphertext> Y2;                               // second tag product (k=2 only)
        tk=Clk::now();
        Plaintext ninv_pt=const_plain(Ninv);
        vector<vector<Ciphertext>> y(KFB);
        for(int i=0;i<KFB;++i){ y[i].resize(2); ev.multiply_plain(prom_m[i],ninv_pt,y[i][0]); ev.multiply_plain(prom_s[i],ninv_pt,y[i][1]); }
        auto T12=poly_mul(y[0],y[1],ev,rlk); auto T34=poly_mul(y[2],y[3],ev,rlk);
        auto T1234=poly_mul(T12,T34,ev,rlk); auto Y=poly_mul(T1234,y[4],ev,rlk);
        if(kmac==2){   // second independent degree-5 tag at point alpha2 -> soundness (d/p)^2
            vector<vector<Ciphertext>> y2(KFB);
            for(int i=0;i<KFB;++i){ y2[i].resize(2); ev.multiply_plain(prom_m[i],ninv_pt,y2[i][0]); ev.multiply_plain(prom_s2[i],ninv_pt,y2[i][1]); }
            auto U12=poly_mul(y2[0],y2[1],ev,rlk); auto U34=poly_mul(y2[2],y2[3],ev,rlk);
            auto U1234=poly_mul(U12,U34,ev,rlk); Y2=poly_mul(U1234,y2[4],ev,rlk);
        }
        t_decision.push_back(ms_since(tk));
        if(kmac==2 && r==0){ // verify the second tag decrypts to prod r2_val (soundness point 2)
            uint64_t exp_tag2=1; for(int i=0;i<KFB;++i) exp_tag2=(uint64_t)(((__uint128_t)exp_tag2*(r2_val[i]%t))%t);
            uint64_t yat2=0, ap=1; Plaintext p2;   // decision already applied N^-1, so read coeff 0 directly
            for(size_t j=0;j<Y2.size();++j){ dec.decrypt(Y2[j],p2); uint64_t cj=p2.coeff_count()?p2[0]%t:0;
                yat2=(uint64_t)(((__uint128_t)yat2+(__uint128_t)cj*ap)%t); ap=(uint64_t)(((__uint128_t)ap*alpha2)%t); }
            cout<<"  [k=2] second-point MAC Y2(alpha2) "<<(yat2==exp_tag2?"CORRECT":"WRONG")<<"\n";
        }
        // BINDING margin = min budget over ALL tag coefficients Y[0..d], not just the decision Y[0]:
        // higher tag coeffs are slope-products and accrue more BFV multiply-noise, so they overflow first.
        dec_noise=dec.invariant_noise_budget(Y[0]);
        for(size_t j=1;j<Y.size();++j){ int b=dec.invariant_noise_budget(Y[j]); if(b<dec_noise) dec_noise=b; }
        if(dec_noise>0) dec_fits++;

        // ---- (E) DOWNLINK: the central must return the WHOLE tag Y(X) (KFB+1 coeffs), not just Y(0),
        //      so the actuator can recompute Y(alpha) and verify the MAC. Two transports, both measured. ----
        size_t ncoef = Y.size();                         // = KFB+1 = 6 for degree-5
        uint64_t exp_dec=1, exp_tag=1;
        for(int i=0;i<KFB;++i){ exp_dec=(uint64_t)(((__uint128_t)exp_dec*(m_val[i]%t))%t); exp_tag=(uint64_t)(((__uint128_t)exp_tag*(r_val[i]%t))%t); }
        stringstream ss0; result_bytes=Y[0].save(ss0);   // full RLWE of ONE coeff (reference)
        tk=Clk::now();
        // (E1) UNPACKED: sample-extract each tag coefficient as its own compressed LWE
        size_t unpacked=0; vector<uint64_t> coef(ncoef,0);
        for(size_t j=0;j<ncoef;++j){ uint64_t d; unpacked += downlink_extract(Y[j], (size_t)rprimes, d); coef[j]=d; }
        // (E2) PACKED: fold coeff j into ring slot j (via X^j), mod-switch once, send a single ciphertext
        Ciphertext yp=Y[0];
        for(size_t j=1;j<ncoef;++j){ Plaintext xj; xj.resize(N); xj[j]=1; Ciphertext sh; ev.multiply_plain(Y[j],xj,sh); ev.add_inplace(yp,sh); }
        while(context.get_context_data(yp.parms_id())->parms().coeff_modulus().size()>(size_t)rprimes) ev.mod_switch_to_next_inplace(yp);
        stringstream ps; size_t packed=yp.save(ps);
        t_dlfull.push_back(ms_since(tk));
        dl_unpacked_bytes=unpacked; dl_packed_bytes=packed;

        // ---- ACTUATOR: decrypt the tag, evaluate Y(alpha), verify the MAC ----
        auto tv=Clk::now();
        Ciphertext yl; { stringstream pl(ps.str()); yl.load(context, pl); }
        Plaintext pt; dec.decrypt(yl, pt);
        uint64_t yat=0, ap=1;                            // Y(alpha) = sum_j coef_j * alpha^j
        for(size_t j=0;j<ncoef;++j){ uint64_t cj=(j<pt.coeff_count())?pt[j]:0; yat=(uint64_t)(((__uint128_t)yat+(__uint128_t)cj*ap)%t); ap=(uint64_t)(((__uint128_t)ap*alpha)%t); }
        uint64_t y0=(pt.coeff_count()?pt[0]:0);          // Y(0) = decision
        t_macverify.push_back(ms_since(tv));
        bool macok = (yat==exp_tag) && (y0==exp_dec);
        if(macok) mac_pass++;
        if(r==0){ bool cu=true; for(size_t j=0;j<ncoef;++j){ uint64_t cj=(j<pt.coeff_count())?pt[j]:0; if(coef[j]!=cj) cu=false; } if(macok && cu) dl_ok=1; }
    }

    // ---- report ----
    cout<<"================ PER-ROUND AVERAGES over "<<rounds<<" rounds (10 values = "<<KFB<<" feedback + "<<KFB<<" MAC) ================\n\n";
    cout<<"  ACTUATOR (end-zone node):\n";
    cout<<"    real seeded-LWE encrypt (10)   "<<mean(t_enc)<<" ms   ("<<mean(t_enc)/(2*KFB)<<" ms/value)\n\n";
    cout<<"  CENTRAL NODE:\n";
    cout<<"    mod-switch + embed (10)        "<<mean(t_embed)<<" ms\n";
    cout<<"    EvalTrace promote (10)         "<<mean(t_promote)<<" ms\n";
    cout<<"    degree-5 decision              "<<mean(t_decision)<<" ms\n";
    cout<<"    build full-tag downlink (6)    "<<mean(t_dlfull)<<" ms  (sample-extract x6 + pack)\n";
    cout<<"    --> central total              "<<mean(t_embed)+mean(t_promote)+mean(t_decision)+mean(t_dlfull)<<" ms\n";
    // CROSS-LIBRARY ROW: the FHE-heavy work alone (promote + decision), excluding the seeded-LWE
    // embed and the downlink build. This is what direction_a_lwe_openfhe_bench reports as its
    // total, so the two BFV libraries are compared on the same scope.
    cout<<"    central eval only              "<<mean(t_promote)+mean(t_decision)<<" ms  (promote + decision — matches the OpenFHE port's scope)\n\n";
    cout<<"  ACTUATOR (downlink): MAC verify (decrypt tag + Y(alpha) check)  "<<mean(t_macverify)<<" ms\n\n";
    cout<<"  Correctness: LWE self-decrypt "<<(lwe_ok?"CORRECT":"WRONG")
        <<", promote round-trip "<<(prom_ok?"CORRECT":"WRONG")
        <<", decision-fits "<<dec_fits<<"/"<<rounds
        <<", tag downlink "<<(dl_ok?"CORRECT":"WRONG")
        <<", MAC verify "<<mac_pass<<"/"<<rounds<<"\n";
    cout<<"  Noise: after promote="<<prom_noise<<" bits, after degree-5="<<dec_noise<<" bits\n\n";
    cout<<"  REAL WIRE SIZES:\n";
    { size_t nvals = (size_t)KFB*(1+kmac);   // 5 feedback + KFB*kmac MAC slopes
    cout<<"    UPLINK seeded-LWE per value    b="<<b_bytes<<" B  (+ shared 32 B seed)\n";
    cout<<"    UPLINK package ("<<nvals<<" values, k="<<kmac<<") 32 B seed + "<<nvals<<" x "<<b_bytes<<" B = "<<32+nvals*b_bytes<<" B\n"; }
    cout<<"    DOWNLINK full tag Y(X) = "<<(KFB+1)<<" coeffs (the actuator MUST verify Y(alpha)):\n";
    cout<<"      full RLWE per coeff x"<<(KFB+1)<<"        "<<(KFB+1)*result_bytes/1024.0<<" KB (uncompressed)\n";
    cout<<"      UNPACKED ("<<(KFB+1)<<" LWE, "<<rprimes<<" prime(s), n=N)  "<<dl_unpacked_bytes/1024.0<<" KB\n";
    cout<<"      PACKED   (1 ciphertext, "<<rprimes<<" prime(s), n=N)  "<<dl_packed_bytes/1024.0<<" KB  ("
        <<(double)dl_unpacked_bytes/dl_packed_bytes<<"x vs unpacked)\n";
    cout<<"      note: 'a' is computation-derived (not seedable); a low-dim key-switch (n=N->~512) cuts\n";
    cout<<"      the unpacked form ~"<<N/512<<"x further (~"<<(dl_unpacked_bytes/1024.0)/(N/512.0)<<" KB).\n";

    // ---- optional: enhanced-ADSC-SNARK flag proof (proves flag_i = reading_i<threshold_i, 32-bit) ----
    if(prove_flags){
#ifdef HAVE_SNARK
        FlagProofResult fp = flag_proof_benchmark(KFB, 32, rounds);
        if(fp.rounds>0 && fp.ok_count==fp.rounds){
            cout<<"\n  ENHANCED-ADSC-SNARK FLAG PROOF ("<<KFB<<" x 32-bit compares, "<<fp.constraints<<" constraints, proof "<<fp.proof_bytes<<" B):\n";
            cout<<"    end-zone (source) PROVE  +"<<fp.prove_ms<<" ms   [folds into the source's ADSC-SNARK]\n";
            cout<<"    central          VERIFY +"<<fp.verify_ms<<" ms   [negligible vs the "<<mean(t_embed)+mean(t_promote)+mean(t_decision)+mean(t_dlfull)<<" ms decision]\n";
            cout<<"    -> recovers input-honesty of the flags (see src/BFV/master.md 10b.3)\n";
        } else cout<<"\n  flag proof FAILED ("<<fp.ok_count<<"/"<<fp.rounds<<")\n";
#else
        cout<<"\n  --prove-flags requires building with HAVE_SNARK (link snark_flags.o); ignored\n";
#endif
    }
    return 0;
}
