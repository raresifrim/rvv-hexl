# mk/seal-bench.mk — build the IPCEI SEAL benches against ONE SEAL install.
# Invoked by `make seal-bench` (the selected build) and by `make bench`
# (every SEAL build of the ISA); can also be run by hand:
#
#   make -f mk/seal-bench.mk SEAL_INSTALL=build/seal/rvv/rvvhexl/install \
#        OUTDIR=build/bench-seal/rvv/rvvhexl
#
# The backend is read from the install's own seal/util/config.h
# (SEAL_USE_INTEL_HEXL), never from a directory name.
#
# The sources in bench/seal/ are verbatim copies from ZKP+FHE Research/cpp_benches
# (unchanged since 899e7b8, identical at 37a6844 and 9d51ffd), so numbers are directly
# comparable with results/benchmarks-*.json there (bfv_seal_deployable,
# bfv_seal_pinned_60bit, ntt_seal). Same flags as the suite's CMake bench builds:
# -O3 + the ISA flags, static libseal, GMP for the BFV benches.
#
# Both SEAL builds get byte-identical bench sources and flags; only the linked SEAL
# (and, with rvv-hexl, the HEXL behind it) changes.

include $(dir $(lastword $(MAKEFILE_LIST)))config.mk

SEAL_INSTALL ?= $(SEAL_PREFIX)
HEXL_PREFIX  ?= $(RVV_HEXL_PREFIX)
OUTDIR       ?= $(ROOT)/build/bench-seal/$(ISA)/$(notdir $(patsubst %/,%,$(dir $(SEAL_INSTALL))))
SRC          := $(ROOT)/bench/seal

SEAL_INC_DIR := $(firstword $(wildcard $(SEAL_INSTALL)/include/SEAL-*))
SEAL_LIB     := $(firstword $(wildcard $(SEAL_INSTALL)/lib/libseal-*.a))
ifeq ($(SEAL_LIB),)
  $(error no SEAL install at $(SEAL_INSTALL) (make seal ...))
endif
SEAL_WITH_HEXL := $(shell grep -q '^\#define SEAL_USE_INTEL_HEXL' $(SEAL_INC_DIR)/seal/util/config.h && echo ON || echo OFF)

# libseal is static and does not embed HEXL: the benches link rvv-hexl themselves.
BENCH_INC := -I$(SEAL_INC_DIR)
BENCH_LIB := $(SEAL_LIB)
ifeq ($(SEAL_WITH_HEXL),ON)
  BENCH_INC += -I$(HEXL_PREFIX)/include
  BENCH_LIB += $(HEXL_PREFIX)/lib/libhexl.a
endif

ifeq ($(HOST_OS),Darwin)
  GMP_INC := -I$(BREW_PREFIX)/include
  GMP_LIB := -L$(BREW_PREFIX)/lib -lgmpxx -lgmp
else
  GMP_INC :=
  GMP_LIB := -lgmpxx -lgmp
endif

BENCH_CXXFLAGS := -std=c++17 -O3 -DNDEBUG $(ISA_FLAGS) -w
BINS := seal_ntt_bench direction_a_lwe_bench direction_a_lwe_bench_30bit

all: $(addprefix $(OUTDIR)/,$(BINS))
	@echo "IPCEI SEAL benches (rvv-hexl=$(SEAL_WITH_HEXL)) -> $(OUTDIR)"

$(OUTDIR)/seal_ntt_bench: $(SRC)/ntt/seal_ntt_bench.cpp
	@mkdir -p $(OUTDIR)
	$(CXX) $(BENCH_CXXFLAGS) $(BENCH_INC) $< -o $@ $(BENCH_LIB) -lpthread

$(OUTDIR)/direction_a_lwe_bench: $(SRC)/bfv/direction_a_lwe_bench.cpp
	@mkdir -p $(OUTDIR)
	$(CXX) $(BENCH_CXXFLAGS) $(BENCH_INC) $(GMP_INC) $< -o $@ $(BENCH_LIB) $(GMP_LIB) -lpthread

$(OUTDIR)/direction_a_lwe_bench_30bit: $(SRC)/bfv/direction_a_lwe_bench_30bit.cpp
	@mkdir -p $(OUTDIR)
	$(CXX) $(BENCH_CXXFLAGS) $(BENCH_INC) $(GMP_INC) $< -o $@ $(BENCH_LIB) $(GMP_LIB) -lpthread

.PHONY: all
