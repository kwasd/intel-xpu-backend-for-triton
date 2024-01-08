#include <pybind11/pybind11.h>
namespace py = pybind11;

void init_triton_env_vars(pybind11::module &m);
void init_triton_runtime(pybind11::module &&m);
void init_triton_ir(pybind11::module &&m);
void init_triton_interpreter(pybind11::module &&m);
void init_triton_translation(pybind11::module &&m);

PYBIND11_MODULE(libintel_xpu_backend_for_triton, m) {
  m.doc() = "Python bindings to the C++ Intel XPU Backend for Triton API";
  init_triton_env_vars(m);
  init_triton_runtime(m.def_submodule("runtime"));
  init_triton_ir(m.def_submodule("ir"));
  init_triton_interpreter(m.def_submodule("interpreter"));
  init_triton_translation(m.def_submodule("translation"));
}
