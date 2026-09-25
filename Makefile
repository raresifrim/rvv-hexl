# rvv-hexl — RISC-V (RVV 1.0) port of Intel HEXL, drop-in for OpenFHE.
#
# ---- rvv-hexl (the library) ------------------------------------------------
#   make rvv-hexl             libhexl.a (+ libhexl.so)   [default goal]
#   make rvv-hexl-test        build + run the tests (RVV path, then native path)   alias: test
#   make rvv-hexl-bench       build bench-hexl (Google Benchmark)
#   make rvv-hexl-install     headers, libs, CMake + pkg-config files -> PREFIX     alias: install
#       ISA=rvv|scalar (auto-detected on RISC-V, see mk/detect-riscv.sh)  BUILD=release|debug  PREFIX=...
#       HEXL_SHARED_LIB / HEXL_TESTING / HEXL_BENCHMARK=ON|OFF   (upstream option names)
#       HEXL_IMPL=intel INTEL_HEXL_PREFIX=...   same tests/benches against upstream Intel HEXL
#
# ---- OpenFHE ------------------------------------------------------------------
#   make openfhe              build OpenFHE v1.5.1
#       NATIVE_SIZE=64|32     default: the CPU word size (64 on riscv64)
#       WITH_RVV_HEXL=OFF|ON  ON = openfhe-hexl backend linked against rvv-hexl
#                             (builds + installs rvv-hexl first; needs NATIVE_SIZE=64)
#   make openfhe-check        OpenFHE's own unit tests for the selected build
#   make openfhe-bench        IPCEI benches against the selected build
#   make openfhe-all          the comparison set: n64, n32, n64 + rvv-hexl
#   make openfhe-list         what is built for this ISA
#
# ---- everything ---------------------------------------------------------------
#   make bench                rvv-hexl-bench + IPCEI benches for every OpenFHE build + tools
#   make run-bench            bench/run.sh (see bench/run.sh --help)
#   make todo | info | reconfigure | clean | distclean
#   make rvv-intrinsics-doc   RVA23 RVV intrinsics reference + index into docs/rvv_intrinsics
#
#   make ISA=scalar ...       any of the above, RISC-V without V (the no-RVV baseline)
#   make CROSS=riscv64-unknown-elf- RISCV_MARCH=rv64gcv RUN="spike --isa=rv64gcv pk" test
#
# See README.md and docs/PORTING_GUIDE.md.

include mk/config.mk

.DEFAULT_GOAL := rvv-hexl
.PHONY: rvv-hexl rvv-hexl-test rvv-hexl-bench rvv-hexl-install rvv-hexl-uninstall \
        test install gbench \
        openfhe openfhe-check openfhe-bench openfhe-all openfhe-list \
        bench bench-openfhe-all tools run-bench todo info reconfigure format clean distclean \
        rvv-intrinsics-doc

# ===========================================================================
# rvv-hexl
# ===========================================================================
LIB_SRCS := $(sort $(wildcard src/*/*.cpp))
LIB_OBJS := $(patsubst src/%.cpp,$(OBJDIR)/lib/%.o,$(LIB_SRCS))
LIB_A    := $(LIBDIR)/libhexl.a
LIB_SO   := $(LIBDIR)/libhexl.$(SHLIB_EXT)
LIB_CPPFLAGS := -Iinclude -Isrc -DHEXL_BUILD_FLAGS='"$(strip $(OPT_FLAGS) $(ISA_FLAGS))"'

TEST_SRCS := $(sort $(wildcard test/*.cpp))
TEST_OBJS := $(patsubst test/%.cpp,$(OBJDIR)/test/%.o,$(TEST_SRCS))
TEST_BIN  := $(BINDIR)/hexl-tests
STRICT    ?=
TEST_ARGS ?= $(if $(STRICT),--strict,)

BENCH_HEXL_SRCS := $(sort $(wildcard bench/hexl/*.cpp))
BENCH_HEXL_OBJS := $(patsubst bench/hexl/%.cpp,$(OBJDIR)/bench/hexl/%.o,$(BENCH_HEXL_SRCS))
BENCH_HEXL_BIN  := $(BINDIR)/bench-hexl

# Rebuild everything in $(BUILDDIR) when the compiler or flags change (e.g. a
# different -march after `make reconfigure` or on another board): make itself
# only compares timestamps. The stamp is rewritten only when its content changes.
FLAGS_STAMP   := $(BUILDDIR)/.build-flags
CURRENT_FLAGS := $(CXX) $(CXXFLAGS) $(HEXL_IMPL) $(INTEL_HEXL_PREFIX)
FLAGS_CHANGED := $(shell mkdir -p $(BUILDDIR) && \
  if [ "$$(cat $(FLAGS_STAMP) 2>/dev/null)" != "$(CURRENT_FLAGS)" ]; then \
    echo "$(CURRENT_FLAGS)" > $(FLAGS_STAMP); echo yes; fi)

ifeq ($(HEXL_IMPL),rvv)
  HEXL_CPPFLAGS := -Iinclude
  HEXL_DEPS     := $(LIB_A)
  HEXL_LDLIBS   := $(LIB_A)
  LIB_TARGETS   := $(LIB_A) $(if $(filter 1,$(SHARED)),$(LIB_SO))
else ifeq ($(HEXL_IMPL),intel)
  HEXL_CPPFLAGS := -I$(INTEL_HEXL_PREFIX)/include
  HEXL_DEPS     :=
  HEXL_LDLIBS   := -L$(INTEL_HEXL_PREFIX)/lib -L$(INTEL_HEXL_PREFIX)/lib64 \
                   $(call RPATH,$(INTEL_HEXL_PREFIX)/lib) -lhexl
  LIB_TARGETS   :=
else
  $(error HEXL_IMPL must be rvv or intel)
endif

rvv-hexl: $(LIB_TARGETS) \
          $(if $(filter ON,$(call onoff,$(HEXL_TESTING))),$(TEST_BIN)) \
          $(if $(filter ON,$(call onoff,$(HEXL_BENCHMARK))),$(BENCH_HEXL_BIN))
ifeq ($(HEXL_IMPL),intel)
	@echo "HEXL_IMPL=intel: library comes from $(INTEL_HEXL_PREFIX)"
endif

$(OBJDIR)/lib/%.o: src/%.cpp $(FLAGS_STAMP)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(LIB_CPPFLAGS) -c $< -o $@

$(LIB_A): $(LIB_OBJS)
	@mkdir -p $(dir $@)
	@rm -f $@
	$(AR) rcs $@ $^

$(LIB_SO): $(LIB_OBJS)
	@mkdir -p $(dir $@)
	$(CXX) -shared $(SONAME_FLAG) $(LDFLAGS) -o $@ $^ $(LDLIBS)

# ---- tests (self-contained harness, no GoogleTest dependency) --------------
$(OBJDIR)/test/%.o: test/%.cpp $(FLAGS_STAMP)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(HEXL_CPPFLAGS) -Itest -c $< -o $@

$(TEST_BIN): $(TEST_OBJS) $(HEXL_DEPS)
	@mkdir -p $(dir $@)
	$(CXX) $(LDFLAGS) -o $@ $(TEST_OBJS) $(HEXL_LDLIBS) $(LDLIBS)

# Runs twice on an RVV build: dispatch as-is, then forced onto the native path,
# so both implementations are checked against the same oracles.
rvv-hexl-test: $(TEST_BIN)
	@echo "==> $(VARIANT): default dispatch"
	$(RUN) $(TEST_BIN) $(TEST_ARGS)
ifeq ($(ISA)$(HEXL_IMPL)$(RUN),rvvrvv)
	@echo "==> $(VARIANT): HEXL_DISABLE_RVV=1 (native path)"
	HEXL_DISABLE_RVV=1 $(RUN) $(TEST_BIN) $(TEST_ARGS)
endif

test: rvv-hexl-test

# ---- HEXL microbenchmarks (Google Benchmark) -------------------------------
$(OBJDIR)/bench/hexl/%.o: bench/hexl/%.cpp $(FLAGS_STAMP)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(HEXL_CPPFLAGS) $(GBENCH_CXXFLAGS) -Itest -c $< -o $@

$(BENCH_HEXL_BIN): $(BENCH_HEXL_OBJS) $(HEXL_DEPS)
	@mkdir -p $(dir $@)
	$(CXX) $(LDFLAGS) -o $@ $(BENCH_HEXL_OBJS) $(HEXL_LDLIBS) $(GBENCH_LDLIBS) $(LDLIBS)

rvv-hexl-bench: $(BENCH_HEXL_BIN)

gbench:
	third_party/gbench.sh "$(CXX)" "$(ROOT)/third_party/install/gbench"

# ---- install (what OpenFHE's INTEL_HEXL_PREBUILT=ON consumes) --------------
HEXL_VERSION := 1.2.6

rvv-hexl-install: $(LIB_TARGETS)
ifneq ($(HEXL_IMPL),rvv)
	$(error rvv-hexl-install needs HEXL_IMPL=rvv)
endif
	@mkdir -p $(PREFIX)/include $(PREFIX)/lib/cmake/hexl-$(HEXL_VERSION) $(PREFIX)/lib/pkgconfig
	cp -R include/hexl $(PREFIX)/include/
	cp $(LIB_A) $(PREFIX)/lib/
ifeq ($(SHARED),1)
	cp $(LIB_SO) $(PREFIX)/lib/
endif
	sed -e 's|@HEXL_LIB_NAME@|libhexl.$(if $(filter 1,$(SHARED)),$(SHLIB_EXT),a)|' \
	    -e 's|@HEXL_LIB_TYPE@|$(if $(filter 1,$(SHARED)),SHARED,STATIC)|' \
	    -e 's|@HEXL_VERSION@|$(HEXL_VERSION)|' \
	    cmake/HEXLConfig.cmake.in > $(PREFIX)/lib/cmake/hexl-$(HEXL_VERSION)/HEXLConfig.cmake
	sed -e 's|@HEXL_VERSION@|$(HEXL_VERSION)|' \
	    cmake/HEXLConfigVersion.cmake.in > $(PREFIX)/lib/cmake/hexl-$(HEXL_VERSION)/HEXLConfigVersion.cmake
	sed -e 's|@PREFIX@|$(PREFIX)|' -e 's|@HEXL_VERSION@|$(HEXL_VERSION)|' \
	    cmake/hexl.pc.in > $(PREFIX)/lib/pkgconfig/hexl.pc
	@echo "installed rvv-hexl ($(VARIANT)) -> $(PREFIX)"

install: rvv-hexl-install

rvv-hexl-uninstall:
	rm -rf $(PREFIX)/include/hexl $(PREFIX)/lib/libhexl.* \
	       $(PREFIX)/lib/cmake/hexl-$(HEXL_VERSION) $(PREFIX)/lib/pkgconfig/hexl.pc

# ===========================================================================
# OpenFHE  (third_party/openfhe.sh does the CMake work)
# ===========================================================================
OFHE_ENV = CXX="$(CXX)" CC="$(CC)" JOBS=$(JOBS) ISA_FLAGS="$(ISA_FLAGS)" \
           OPENFHE_TAG=$(OPENFHE_TAG) OPENFHE_HEXL_TAG=$(OPENFHE_HEXL_TAG) \
           OPENFHE_DIR="$(OPENFHE_DIR)" OPENFHE_PREFIX="$(OPENFHE_PREFIX)" \
           NATIVE_SIZE=$(NATIVE_SIZE) WITH_RVV_HEXL=$(WITH_RVV_HEXL) \
           HEXL_PREFIX="$(RVV_HEXL_PREFIX)" \
           OPENFHE_BENCHMARKS=$(call onoff,$(OPENFHE_BENCHMARKS)) \
           OPENFHE_UNITTESTS=$(call onoff,$(OPENFHE_UNITTESTS))

ifneq ($(filter openfhe,$(MAKECMDGOALS)),)
  ifeq ($(WITH_RVV_HEXL),ON)
    ifneq ($(NATIVE_SIZE),64)
      $(error WITH_RVV_HEXL=ON needs NATIVE_SIZE=64: OpenFHE's HEXL backend casts coefficient vectors to uint64_t*; the e32 path lives inside rvv-hexl)
    endif
  endif
endif

# With WITH_RVV_HEXL=ON, (re)install rvv-hexl first: same ISA and BUILD.
openfhe: $(if $(filter ON,$(WITH_RVV_HEXL)),rvv-hexl-install)
	$(OFHE_ENV) third_party/openfhe.sh build

openfhe-check:
	$(OFHE_ENV) third_party/openfhe.sh check

openfhe-bench:
	$(MAKE) --no-print-directory -f mk/openfhe-bench.mk \
	  OFHE_PREFIX="$(OPENFHE_PREFIX)" HEXL_PREFIX="$(RVV_HEXL_PREFIX)" \
	  OUTDIR="$(ROOT)/build/bench-openfhe/$(ISA)/$(OPENFHE_NAME)"

openfhe-all:
	$(MAKE) --no-print-directory openfhe NATIVE_SIZE=64 WITH_RVV_HEXL=OFF
	$(MAKE) --no-print-directory openfhe NATIVE_SIZE=32 WITH_RVV_HEXL=OFF
	$(MAKE) --no-print-directory openfhe NATIVE_SIZE=64 WITH_RVV_HEXL=ON

openfhe-list:
	@for d in $(OPENFHE_ROOT)/*/install; do \
	  [ -d "$$d" ] || continue; \
	  cfg="$$d/include/openfhe/core/config_core.h"; \
	  n=$$(sed -n 's/^#define NATIVEINT \([0-9]*\).*/\1/p' "$$cfg"); \
	  h=$$(grep -q '^#define WITH_INTEL_HEXL' "$$cfg" && echo "rvv-hexl" || echo "stock"); \
	  printf '  %-24s NATIVE_SIZE=%-3s %s\n' "$$(basename $$(dirname $$d))" "$$n" "$$h"; \
	done

# ===========================================================================
# Benchmarks
# ===========================================================================
# IPCEI benches for EVERY OpenFHE build of this ISA (what bench/run.sh expects)
bench-openfhe-all:
	@for d in $(OPENFHE_ROOT)/*/install; do \
	  [ -d "$$d" ] || { echo "no OpenFHE build under $(OPENFHE_ROOT) (make openfhe)"; break; }; \
	  name=$$(basename $$(dirname $$d)); \
	  $(MAKE) --no-print-directory -f mk/openfhe-bench.mk OFHE_PREFIX="$$d" \
	    HEXL_PREFIX="$(ROOT)/build/$(ISA)-$$(case $$name in *-debug) echo debug;; *) echo release;; esac)/install" \
	    OUTDIR="$(ROOT)/build/bench-openfhe/$(ISA)/$$name" || exit 1; \
	done

tools: $(ROOT)/build/tools/ailaunch

$(ROOT)/build/tools/ailaunch: bench/tools/ailaunch.c
	@mkdir -p $(dir $@)
	$(CC) -O2 -o $@ $<

bench: rvv-hexl-bench bench-openfhe-all $(if $(filter riscv64,$(HOST_ARCH)),tools)

run-bench:
	bench/run.sh

# ===========================================================================
# Housekeeping
# ===========================================================================
todo:
	@grep -rn --include='*.cpp' --include='*.hpp' 'HEXL_NOT_IMPLEMENTED();' src \
	  | sort -t: -k1,1 -k2,2n \
	  | sed -e 's/:[[:space:]]*HEXL_NOT_IMPLEMENTED();//' \
	  | awk -F: '{print $$1":"$$2}' \
	  | while IFS=: read f l; do \
	      fn=$$(awk -v L=$$l 'NR<=L && /^[a-zA-Z].*\(/ {s=$$0} NR==L {print s}' $$f | sed 's/(.*//; s/.* //'); \
	      printf '  %-40s %s:%s\n' "$$fn" "$$f" "$$l"; \
	    done
	@printf '%s stubs left\n' "$$(grep -rn --include='*.cpp' --include='*.hpp' 'HEXL_NOT_IMPLEMENTED();' src | wc -l | tr -d ' ')"

info:
	@echo "host            : $(HOST_OS)/$(HOST_ARCH)"
	@echo "target          : $(TARGET_TRIPLE) ($(TARGET_ARCH), $(CPU_WORD_BITS)-bit)$(if $(BAREMETAL), [bare metal])"
	@echo "CXX             : $(CXX)  ($(shell $(CXX) --version 2>/dev/null | head -1))"
	@echo "ISA / BUILD     : $(ISA) / $(BUILD)   -> $(ISA_FLAGS) $(OPT_FLAGS)"
ifeq ($(TARGET_ARCH),riscv64)
	@echo "RISC-V check    : requested $(ISA_REQUEST), $(RV_DETECT_MODE) build -> $(RVV_DETECT_ISA)$(if $(RVV_DETECT_NOTE), (fallback: $(RVV_DETECT_NOTE)))"
	@echo "  rvv march     : $(if $(RVV_DETECT_MARCH),$(RVV_DETECT_MARCH),-)"
	@echo "  scalar march  : $(RVV_DETECT_SCALAR_MARCH)"
	@echo "  board         : $(if $(RVV_DETECT_HOST),$(RVV_DETECT_HOST),n/a (cross build))"
endif
	@echo "--- rvv-hexl"
	@echo "HEXL_IMPL       : $(HEXL_IMPL)$(if $(filter intel,$(HEXL_IMPL)), ($(INTEL_HEXL_PREFIX)))"
	@echo "build dir       : $(BUILDDIR)"
	@echo "install PREFIX  : $(PREFIX)"
	@echo "HEXL_SHARED_LIB : $(call onoff,$(HEXL_SHARED_LIB))   HEXL_TESTING: $(call onoff,$(HEXL_TESTING))   HEXL_BENCHMARK: $(call onoff,$(HEXL_BENCHMARK))"
	@echo "google bench    : $(if $(GBENCH_PREFIX),$(GBENCH_PREFIX),system)"
	@echo "--- openfhe"
	@echo "version         : $(OPENFHE_TAG) (+ openfhe-hexl $(OPENFHE_HEXL_TAG) when WITH_RVV_HEXL=ON)"
	@echo "NATIVE_SIZE     : $(NATIVE_SIZE)   WITH_RVV_HEXL: $(WITH_RVV_HEXL)$(if $(filter ON,$(WITH_RVV_HEXL)), (rvv-hexl from $(RVV_HEXL_PREFIX)))"
	@echo "benchmarks/tests: $(call onoff,$(OPENFHE_BENCHMARKS)) / $(call onoff,$(OPENFHE_UNITTESTS))"
	@echo "dir             : $(OPENFHE_DIR)"

# RVV C intrinsics reference for the whole RVA23 vector ISA (not committed, see
# .gitignore): rvv-intrinsic-doc v1.0-ratified + the vector-crypto and BF16
# chapters from main, plus INDEX.md and a grep-able rva23-intrinsics.tsv that
# marks, per intrinsic, its extension, RVA23 status, K3 support and whether the
# RISC-V compiler implements it. See docs/tools/fetch-rvv-intrinsics.sh.
rvv-intrinsics-doc:
	$(if $(filter riscv64,$(TARGET_ARCH)),PROBE_CC="$(CC)") docs/tools/fetch-rvv-intrinsics.sh

# Forget the cached RISC-V toolchain/board checks (see mk/detect-riscv.sh)
reconfigure:
	rm -rf $(ROOT)/build/config
	@echo "RISC-V checks will run again on the next make"

format:
	clang-format -i $$(find include src test bench/hexl -name '*.hpp' -o -name '*.cpp')

clean:
	rm -rf $(BUILDDIR)

distclean:
	rm -rf build third_party/src third_party/install

-include $(LIB_OBJS:.o=.d) $(TEST_OBJS:.o=.d) $(BENCH_HEXL_OBJS:.o=.d)
