// Copyright (C) 2026 IPCEI-NXP A14 team (RISC-V port)
// SPDX-License-Identifier: Apache-2.0
//
// Test runner.
//   hexl-tests                 run everything
//   hexl-tests Ntt Eltwise     run tests whose name contains any of the words
//   hexl-tests --list          list test names
//   hexl-tests --strict        TODO counts as failure (exit code 2)
//   hexl-tests --quick         smaller sizes (for spike, which is ~1000x slower)
//   hexl-tests -v              print every test, not only the non-passing ones
// Exit code: 0 all pass (TODOs allowed unless --strict), 1 any FAIL.

#include <cstring>
#include <exception>
#include <stdexcept>

#include "hexl/hexl.hpp"
#include "oracle.hpp"
#include "test.hpp"

namespace hexltest {
bool g_quick = false;
}

namespace {

const char kTodoPrefix[] = "[rvv-hexl TODO]";

bool Selected(const char* name, const std::vector<const char*>& filters) {
  if (filters.empty()) return true;
  for (const char* f : filters) {
    if (std::strstr(name, f)) return true;
  }
  return false;
}

void PrintBuildInfo() {
  const auto info = intel::hexl::rvv::GetPortInfo();
  std::printf("rvv-hexl: rvv compiled=%d available=%d enabled=%d VLEN=%zu\n",
              info.compiled_with_rvv, info.rvv_available, info.rvv_enabled,
              info.vlen_bits);
  std::printf("            flags: %s\n            compiler: %s\n",
              info.build_flags, info.compiler);
}

}  // namespace

int main(int argc, char** argv) {
  bool strict = false, verbose = false, list = false;
  std::vector<const char*> filters;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--strict")) strict = true;
    else if (!std::strcmp(argv[i], "--quick")) hexltest::g_quick = true;
    else if (!std::strcmp(argv[i], "-v")) verbose = true;
    else if (!std::strcmp(argv[i], "--list")) list = true;
    else filters.push_back(argv[i]);
  }

  auto& cases = hexltest::Registry();
  if (list) {
    for (const auto& c : cases) std::printf("%s\n", c.name);
    return 0;
  }

  PrintBuildInfo();
  int pass = 0, fail = 0, todo = 0;
  for (const auto& c : cases) {
    if (!Selected(c.name, filters)) continue;
    try {
      c.fn();
      ++pass;
      if (verbose) std::printf("  PASS  %s\n", c.name);
    } catch (const hexltest::Failure& f) {
      ++fail;
      std::printf("  FAIL  %s\n        %s\n", c.name, f.message.c_str());
    } catch (const std::logic_error& e) {
      if (std::strncmp(e.what(), kTodoPrefix, sizeof(kTodoPrefix) - 1) == 0) {
        ++todo;
        std::printf("  TODO  %-44s <- %s\n", c.name,
                    e.what() + sizeof(kTodoPrefix));
      } else {
        ++fail;
        std::printf("  FAIL  %s\n        logic_error: %s\n", c.name, e.what());
      }
    } catch (const std::exception& e) {
      ++fail;
      std::printf("  FAIL  %s\n        exception: %s\n", c.name, e.what());
    }
  }
  std::printf("%d passed, %d failed, %d TODO (stub reached)\n", pass, fail,
              todo);
  if (fail) return 1;
  if (strict && todo) return 2;
  return 0;
}
