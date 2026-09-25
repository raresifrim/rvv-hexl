# mk/openfhe-bench.mk — build the IPCEI OpenFHE benches against ONE OpenFHE install.
# Invoked by `make openfhe-bench` (the selected build) and by `make bench`
# (every build of the ISA); can also be run by hand:
#
#   make -f mk/openfhe-bench.mk OFHE_PREFIX=build/openfhe/rvv/n64/install \
#        OUTDIR=build/bench-openfhe/rvv/n64
#
# Word size and backend are read from the install's own config_core.h
# (NATIVEINT, WITH_INTEL_HEXL), never from a directory name.
#
# The sources in bench/openfhe/ are verbatim copies from
# ZKP+FHE Research/cpp_benches (commit 37a6844), so numbers are directly
# comparable with results/benchmarks-*.json there. Only the build system
# differs (make instead of CMake), with the same flags: -O3 + the ISA flags,
# OpenMP, shared OpenFHE, no -DPARALLEL (same as the CMake bench builds).
#
# Every OpenFHE build gets byte-identical bench sources and flags; the only
# thing that changes is which OpenFHE (and, with rvv-hexl, which HEXL) is
# linked, so a delta between builds is attributable to the library alone.

include $(dir $(lastword $(MAKEFILE_LIST)))config.mk

OFHE_PREFIX  ?= $(OPENFHE_PREFIX)
HEXL_PREFIX  ?= $(RVV_HEXL_PREFIX)
OUTDIR       ?= $(ROOT)/build/bench-openfhe/$(ISA)/$(notdir $(patsubst %/,%,$(dir $(OFHE_PREFIX))))
SRC          := $(ROOT)/bench/openfhe

OFHE_CONFIG  := $(OFHE_PREFIX)/include/openfhe/core/config_core.h
ifeq ($(wildcard $(OFHE_CONFIG)),)
  $(error no OpenFHE install at $(OFHE_PREFIX) (make openfhe ...))
endif
OFHE_NATIVEINT := $(shell sed -n 's/^\#define NATIVEINT \([0-9]*\).*/\1/p' $(OFHE_CONFIG))
OFHE_WITH_HEXL := $(shell grep -q '^\#define WITH_INTEL_HEXL' $(OFHE_CONFIG) && echo ON || echo OFF)

OFHE_INC := $(addprefix -I$(OFHE_PREFIX)/include/openfhe,/ /core /pke /binfhe /cereal) \
            -I$(OFHE_PREFIX)/include
OFHE_LIB := -L$(OFHE_PREFIX)/lib -L$(OFHE_PREFIX)/lib64 \
            $(call RPATH,$(OFHE_PREFIX)/lib) $(call RPATH,$(OFHE_PREFIX)/lib64) \
            -lOPENFHEbinfhe -lOPENFHEpke -lOPENFHEcore

# rvv-hexl backend: OpenFHE's installed HEXL headers #include "hexl/hexl.hpp",
# and its .so has a DT_NEEDED on libhexl.so -> both must resolve to rvv-hexl.
ifeq ($(OFHE_WITH_HEXL),ON)
  OFHE_INC += -I$(HEXL_PREFIX)/include
  OFHE_LIB += -L$(HEXL_PREFIX)/lib $(call RPATH,$(HEXL_PREFIX)/lib) -lhexl
endif

ifeq ($(HOST_OS),Darwin)
  SSL_INC := -I$(BREW_PREFIX)/opt/openssl@3/include
  SSL_LIB := -L$(BREW_PREFIX)/opt/openssl@3/lib -lcrypto
else
  SSL_INC :=
  SSL_LIB := -lcrypto
endif

BENCH_CXXFLAGS := -std=c++17 -O3 -DNDEBUG $(ISA_FLAGS) $(OPENMP_CXXFLAGS) -w
BENCH_LDLIBS   := $(OFHE_LIB) $(OPENMP_LDLIBS) -lpthread

# NATIVE_SIZE=32 cannot hold the 49/60-bit NTT and BFV moduli: TFHE benches only
# (same split as the IPCEI suite's build-n32 tree).
ifeq ($(OFHE_NATIVEINT),32)
  BINS := direction_a_tfhe_bench direction_a_tfhe_lut_bench
else
  BINS := openfhe_ntt_bench direction_a_lwe_openfhe_bench \
          direction_a_tfhe_bench direction_a_tfhe_lut_bench
endif

all: $(addprefix $(OUTDIR)/,$(BINS))
	@echo "IPCEI OpenFHE benches (NATIVE_SIZE=$(OFHE_NATIVEINT), rvv-hexl=$(OFHE_WITH_HEXL)) -> $(OUTDIR)"

$(OUTDIR)/openfhe_ntt_bench: $(SRC)/ntt/openfhe_ntt_bench.cpp
	@mkdir -p $(OUTDIR)
	$(CXX) $(BENCH_CXXFLAGS) $(OFHE_INC) $< -o $@ $(BENCH_LDLIBS)

$(OUTDIR)/direction_a_lwe_openfhe_bench: $(SRC)/bfv/direction_a_lwe_openfhe_bench.cpp
	@mkdir -p $(OUTDIR)
	$(CXX) $(BENCH_CXXFLAGS) $(OFHE_INC) $< -o $@ $(BENCH_LDLIBS)

$(OUTDIR)/direction_a_tfhe_bench: $(SRC)/tfhe/direction_a_tfhe_bench.cpp
	@mkdir -p $(OUTDIR)
	$(CXX) $(BENCH_CXXFLAGS) $(OFHE_INC) $(SSL_INC) $< -o $@ $(BENCH_LDLIBS) $(SSL_LIB)

$(OUTDIR)/direction_a_tfhe_lut_bench: $(SRC)/tfhe/direction_a_tfhe_lut_bench.cpp
	@mkdir -p $(OUTDIR)
	$(CXX) $(BENCH_CXXFLAGS) $(OFHE_INC) $< -o $@ $(BENCH_LDLIBS)

.PHONY: all
