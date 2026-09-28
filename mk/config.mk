# mk/config.mk — toolchain, ISA variant and directory layout.
# Included by the top-level Makefile. Every variable can be overridden on the
# command line:  make ISA=scalar BUILD=debug CXX=g++-15 ...
#
# Kept compatible with GNU make 3.81 (the macOS default) so the library and the
# tests can also be developed on a laptop; the real target is Linux/riscv64.

ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)

# onoff(<value>) -> ON or OFF. Accepts ON/OFF, 1/0, yes/no, true/false (CMake style).
onoff = $(if $(filter ON On on 1 YES Yes yes TRUE True true Y y,$(strip $(1))),ON,OFF)

HOST_OS   := $(shell uname -s)
HOST_ARCH := $(shell uname -m)

# ---------------------------------------------------------------------------
# Toolchain
# ---------------------------------------------------------------------------
# CROSS=<prefix>  builds with <prefix>g++ (e.g. CROSS=riscv64-unknown-elf- for
# spike + pk on a laptop, or riscv64-linux-gnu- for a Linux cross build).
CROSS ?=
ifneq ($(CROSS),)
  CXX := $(CROSS)g++
  CC  := $(CROSS)gcc
  AR  := $(CROSS)ar
else ifeq ($(origin CXX),default)
  # Native: on riscv64 prefer the newest GCC (RVA23 needs GCC >= 14).
  ifeq ($(HOST_ARCH),riscv64)
    CXX := $(firstword $(foreach c,g++-16 g++-15 g++-14 g++,$(if $(shell command -v $(c) 2>/dev/null),$(c))))
    CC  := $(subst g++,gcc,$(CXX))
  else
    CXX := g++
  endif
endif
AR ?= ar

TARGET_ARCH := $(if $(CROSS),$(firstword $(subst -, ,$(shell $(CXX) -dumpmachine))),$(HOST_ARCH))
TARGET_TRIPLE := $(shell $(CXX) -dumpmachine 2>/dev/null)
# Bare-metal newlib toolchains (spike + pk): no shared libs, no threads.
BAREMETAL := $(if $(findstring -elf,$(TARGET_TRIPLE)),1,)

# ---------------------------------------------------------------------------
# ISA variant
# ---------------------------------------------------------------------------
#   rvv     RISC-V with V: the port's RVV kernels are compiled in
#   scalar  RISC-V WITHOUT V: exactly the rvv -march minus V (and zv*). The
#           library is pure native C++ and the compiler cannot auto-vectorise.
#           This is the "base RISC-V (no RVV)" data point.
#   native  non-RISC-V hosts (x86-64/arm64): -march/-mcpu=native, for
#           cross-architecture comparison runs and laptop development.
#
# On RISC-V the choice is checked, not assumed (mk/detect-riscv.sh):
#   * the board (native builds): every hart must report V in /proc/cpuinfo, and
#     -march is limited to the extensions all harts report: rva23u64 on an RVA23
#     board such as the K3, the board's own list otherwise. A -march wider than
#     the CPU is a SIGILL anywhere in the library, not just in the RVV kernels.
#   * the toolchain: the compiler must build the RVV C intrinsics v1.0 API the
#     port uses (GCC >= 14, LLVM >= 17).
# If either check fails, the build falls back to ISA=scalar with a warning, even
# when ISA=rvv was requested. The result is cached per toolchain/request in
# build/config/; `make reconfigure` re-runs the checks (e.g. after a compiler
# upgrade). Explicit RISCV_MARCH / RISCV_SCALAR_MARCH skip the -march choice
# (not the checks). RV_DETECT_MODE=native CPUINFO=<file> simulates another
# board when testing the detection itself.
RISCV_MARCH        ?=
RISCV_SCALAR_MARCH ?=

ifeq ($(TARGET_ARCH),riscv64)
  ISA_REQUEST := $(if $(ISA),$(ISA),auto)
  ifeq ($(filter auto rvv scalar,$(ISA_REQUEST)),)
    $(error ISA=$(ISA) is not valid on riscv64; use rvv or scalar)
  endif
  RV_DETECT_MODE ?= $(if $(CROSS),cross,$(if $(filter riscv64,$(HOST_ARCH)),native,cross))
  RV_DETECT_KEY  := $(firstword $(shell echo '$(CXX)|$(RV_DETECT_MODE)|$(ISA_REQUEST)|$(RISCV_MARCH)|$(RISCV_SCALAR_MARCH)|$(CPUINFO)' | cksum))
  RV_DETECT_FILE := $(ROOT)/build/config/riscv-$(RV_DETECT_KEY).mk
  ifeq ($(wildcard $(RV_DETECT_FILE)),)
    RV_DETECT_RUN := $(shell mkdir -p $(ROOT)/build/config && \
      CXX='$(CXX)' MODE=$(RV_DETECT_MODE) ISA_REQUEST=$(ISA_REQUEST) $(if $(CPUINFO),CPUINFO='$(CPUINFO)') \
      USER_MARCH='$(RISCV_MARCH)' USER_SCALAR_MARCH='$(RISCV_SCALAR_MARCH)' \
      $(ROOT)/mk/detect-riscv.sh > $(RV_DETECT_FILE).tmp && mv $(RV_DETECT_FILE).tmp $(RV_DETECT_FILE))
  endif
  include $(RV_DETECT_FILE)
  override ISA := $(RVV_DETECT_ISA)
  ISA_FLAGS := -march=$(if $(filter rvv,$(ISA)),$(RVV_DETECT_MARCH),$(RVV_DETECT_SCALAR_MARCH))
  ifneq ($(RVV_DETECT_NOTE),)
    ifeq ($(MAKELEVEL),0)
      $(warning RVV disabled, building ISA=scalar: $(RVV_DETECT_NOTE). Run `make reconfigure` to re-check.)
    endif
  endif
else
  ISA ?= native
  ifeq ($(filter $(TARGET_ARCH),arm64 aarch64),)
    NATIVE_FLAG := -march=native
  else
    NATIVE_FLAG := -mcpu=native
  endif
  ifeq ($(ISA),native)
    ISA_FLAGS := $(NATIVE_FLAG)
  else ifeq ($(ISA),scalar)
    # Non-RISC-V "scalar" = same ISA with the auto-vectorisers off. Placed after
    # -O3 on the command line (see CXXFLAGS below) or clang silently re-enables them.
    ISA_FLAGS := $(NATIVE_FLAG) -fno-tree-vectorize -fno-tree-slp-vectorize
  else
    $(error ISA=$(ISA) is only valid on riscv64; use native or scalar here)
  endif
endif

# ---------------------------------------------------------------------------
# Build type
# ---------------------------------------------------------------------------
#   release  -O3, HEXL_CHECK compiled out (what OpenFHE links against)
#   debug    -O1 -g, HEXL_DEBUG: every HEXL_CHECK argument check is live and
#            HEXL_VLOG=<level> prints which kernel was dispatched
#   (CMAKE_BUILD_TYPE=Debug/Release is accepted as an alias, like upstream,
#    where Debug is also what turns HEXL_DEBUG on)
ifdef CMAKE_BUILD_TYPE
  BUILD ?= $(if $(filter Debug debug,$(CMAKE_BUILD_TYPE)),debug,release)
endif
BUILD ?= release
ifeq ($(BUILD),release)
  OPT_FLAGS := -O3 -DNDEBUG
else ifeq ($(BUILD),debug)
  OPT_FLAGS := -O1 -g -DHEXL_DEBUG
else
  $(error BUILD must be release or debug)
endif

SANITIZE ?=
ifneq ($(SANITIZE),)
  OPT_FLAGS += -fsanitize=$(SANITIZE) -fno-omit-frame-pointer
  LDFLAGS   += -fsanitize=$(SANITIZE)
endif

# ---------------------------------------------------------------------------
# Which HEXL the tests/benches link against
# ---------------------------------------------------------------------------
#   rvv    this repository (default)
#   intel  an installed upstream Intel HEXL at INTEL_HEXL_PREFIX: builds the SAME
#          test and benchmark sources against it (e.g. on the AMD AVX-512 box),
#          so cross-architecture numbers come from identical code.
HEXL_IMPL ?= rvv
INTEL_HEXL_PREFIX ?= /usr/local

VARIANT  := $(if $(filter intel,$(HEXL_IMPL)),intel-,)$(ISA)-$(BUILD)
BUILDDIR := $(ROOT)/build/$(VARIANT)
OBJDIR   := $(BUILDDIR)/obj
LIBDIR   := $(BUILDDIR)/lib
BINDIR   := $(BUILDDIR)/bin
PREFIX   ?= $(BUILDDIR)/install

# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------
CXXSTD   := -std=c++17
WARN     := -Wall -Wextra -Wno-unused-parameter
PIC      := $(if $(BAREMETAL),,-fPIC)
CXXFLAGS ?=
CXXFLAGS += $(CXXSTD) $(OPT_FLAGS) $(ISA_FLAGS) $(WARN) $(PIC) -MMD -MP
LDFLAGS  ?=
LDLIBS   ?=
ifeq ($(BAREMETAL),)
  LDLIBS += -lpthread
endif

ifeq ($(HOST_OS),Darwin)
  SHLIB_EXT := dylib
  SONAME_FLAG = -install_name $(abspath $(LIBDIR))/$(notdir $@)
  RPATH = -Wl,-rpath,$(1)
else
  SHLIB_EXT := so
  SONAME_FLAG = -Wl,-soname,libhexl.$(SHLIB_EXT)
  RPATH = -Wl,-rpath,$(1)
endif

# ---------------------------------------------------------------------------
# rvv-hexl options, same names as upstream Intel HEXL's CMake options
# ---------------------------------------------------------------------------
#   HEXL_SHARED_LIB  also build libhexl.so (upstream default OFF; ON here because
#                    OpenFHE is built with BUILD_SHARED=ON and so are the benches)
#   HEXL_TESTING     `make rvv-hexl` also builds the test binary   (upstream: ON)
#   HEXL_BENCHMARK   `make rvv-hexl` also builds bench-hexl        (upstream: ON)
# Upstream defaults TESTING/BENCHMARK to ON; here they are OFF so that building
# the library alone stays fast and dependency-free (bench needs Google Benchmark).
HEXL_SHARED_LIB ?= $(if $(BAREMETAL),OFF,ON)
HEXL_TESTING    ?= OFF
HEXL_BENCHMARK  ?= OFF
SHARED := $(if $(filter ON,$(call onoff,$(HEXL_SHARED_LIB))),1,0)

# Running target binaries. Empty natively; for spike: RUN="spike --isa=rv64gcv pk"
RUN ?=

# OpenMP (only the OpenFHE benches need it; the library itself is thread-free)
ifeq ($(HOST_OS),Darwin)
  BREW_PREFIX := $(shell brew --prefix 2>/dev/null)
  OPENMP_CXXFLAGS := -Xpreprocessor -fopenmp -I$(BREW_PREFIX)/opt/libomp/include
  OPENMP_LDLIBS   := -L$(BREW_PREFIX)/opt/libomp/lib -lomp
else
  OPENMP_CXXFLAGS := -fopenmp
  OPENMP_LDLIBS   := -fopenmp
endif

# Google Benchmark (bench/hexl only): system package (apt install
# libbenchmark-dev) or the copy built by `make gbench` into third_party/install.
GBENCH_PREFIX ?= $(if $(wildcard $(ROOT)/third_party/install/gbench/include/benchmark/benchmark.h),$(ROOT)/third_party/install/gbench,)
GBENCH_CXXFLAGS := $(if $(GBENCH_PREFIX),-I$(GBENCH_PREFIX)/include,)
GBENCH_LDLIBS   := $(if $(GBENCH_PREFIX),-L$(GBENCH_PREFIX)/lib $(call RPATH,$(GBENCH_PREFIX)/lib),) -lbenchmark

# ---------------------------------------------------------------------------
# OpenFHE options  (make openfhe NATIVE_SIZE=.. WITH_RVV_HEXL=..)
# ---------------------------------------------------------------------------
#   NATIVE_SIZE     OpenFHE's native integer width: 64 or 32 (128 is accepted by
#                   OpenFHE on some platforms). Default: the target CPU's word size.
#   WITH_RVV_HEXL   ON: OpenFHE + openfhe-hexl overlay linked against this library
#                   (= upstream's -DWITH_INTEL_HEXL=ON -DINTEL_HEXL_PREBUILT=ON).
#                   Works at NATIVE_SIZE=64 and 32 (the 32-bit build uses
#                   rvv-hexl's uint32_t API through a patched overlay, see
#                   third_party/patches/). Default OFF (stock OpenFHE).
#   OPENFHE_BENCHMARKS / OPENFHE_UNITTESTS  build upstream's benchmark suite /
#                   unit tests (default ON / ON only together with rvv-hexl).
# Each combination gets its own directory, so they coexist and can be compared:
#   build/openfhe/<ISA>/n<NATIVE_SIZE>[-rvvhexl[-debug]]/{build,install}
# The "-debug" suffix marks an OpenFHE linked against a BUILD=debug rvv-hexl
# (OpenFHE itself is always Release; the debug lib turns on every HEXL_CHECK).
CPU_WORD_BITS      := $(if $(filter %64 arm64,$(TARGET_ARCH)),64,32)
NATIVE_SIZE        ?= $(CPU_WORD_BITS)
WITH_RVV_HEXL      ?= OFF
override WITH_RVV_HEXL := $(call onoff,$(WITH_RVV_HEXL))
# (OFF on macOS: upstream's bundled Google Benchmark fails its own -Werror
#  configure checks with Apple clang + libomp; Linux/GCC is unaffected)
OPENFHE_BENCHMARKS ?= $(if $(filter Darwin,$(HOST_OS)),OFF,ON)
OPENFHE_UNITTESTS  ?= $(WITH_RVV_HEXL)
OPENFHE_TAG        ?= v1.5.1
OPENFHE_HEXL_TAG   ?= v1.5.1.0
OPENFHE_ROOT       := $(ROOT)/build/openfhe/$(ISA)
OPENFHE_NAME       ?= n$(NATIVE_SIZE)$(if $(filter ON,$(WITH_RVV_HEXL)),-rvvhexl$(if $(filter debug,$(BUILD)),-debug))
OPENFHE_DIR        := $(OPENFHE_ROOT)/$(OPENFHE_NAME)
OPENFHE_PREFIX     ?= $(OPENFHE_DIR)/install
# The rvv-hexl install an rvvhexl OpenFHE (or SEAL) links against: same ISA and BUILD.
RVV_HEXL_PREFIX    := $(ROOT)/build/$(ISA)-$(BUILD)/install
JOBS ?= $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# ---------------------------------------------------------------------------
# Microsoft SEAL options  (make seal WITH_RVV_HEXL=..)
# ---------------------------------------------------------------------------
#   WITH_RVV_HEXL   shared with OpenFHE. ON: SEAL with -DSEAL_USE_INTEL_HEXL=ON
#                   linked against this library. SEAL always uses 64-bit words,
#                   so there is no NATIVE_SIZE axis. Default OFF (stock SEAL).
#   SEAL_TESTS      also build SEAL's own sealtest (needs libgtest-dev). Default OFF.
# Both configurations coexist:
#   build/seal/<ISA>/{stock,rvvhexl[-debug]}/{build,install}
SEAL_TAG    ?= v4.1.2
SEAL_TESTS  ?= OFF
SEAL_ROOT   := $(ROOT)/build/seal/$(ISA)
SEAL_NAME   ?= $(if $(filter ON,$(WITH_RVV_HEXL)),rvvhexl$(if $(filter debug,$(BUILD)),-debug),stock)
SEAL_DIR    := $(SEAL_ROOT)/$(SEAL_NAME)
SEAL_PREFIX ?= $(SEAL_DIR)/install
