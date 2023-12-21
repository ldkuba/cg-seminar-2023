#pragma once

#include <iostream>

#include "test_method.hpp"
#include "pybind11/pybind11.h"
#include "pybind11/embed.h"
#include "pybind11/eigen.h"

#include "py_util.hpp"

namespace py = pybind11;

class ReachForTheSpheres : public TestMethod {
public:
    ReachForTheSpheres()
        : TestMethod("Reach For The Spheres") {}

    std::unique_ptr<TestResults> ReachForTheSpheres::run(const TestInputData& input) override {
        auto results = std::make_unique<TestResults>();

        PY_CALL_START
        auto locals = py::dict();
        locals["sdf_points"] = input.sdf_points;
        locals["sdf_at_points"] = input.sdf_at_points;
        locals["results"] = results.get();

        py::eval_file("src/methods/reach_for_the_spheres.py", py::globals(), locals);
        *results = locals["results"].cast<TestResults>();
        PY_CALL_END

        // === Set test result data ===
        // Output vertices start at (0, 0, 0). Move all vertices to the aabb minimum
        // Eigen::Vector3f delta(input.bounds(0), input.bounds(1), input.bounds(2));
        // results->out_mesh.vertices.rowwise() += delta.transpose();

        // Method name
        results->method = this->name;

        return results;
    }

};
