#pragma once

#include <string>
#include "pybind11/pybind11.h"
#include <iostream>

namespace py = pybind11;

// Surround calls to pybind11 with these macros to catch errors
#define PY_CALL_START try {
#define PY_CALL_END } catch (const py::error_already_set& e) { \
    std::cerr << "Python error: " << e.what() << std::endl; \
    PyErr_Print(); \
    throw std::runtime_error("Python error"); \
}
