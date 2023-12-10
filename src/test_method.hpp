#pragma once

#include <string>

#include <Eigen/Dense>
#include "sdf/sdf.hpp"

using AABB = Eigen::Ref<const Eigen::Matrix<float, 6, 1>>;

struct TriangleMesh {
    sdf::Points vertices;
    sdf::Triangles indices;
};

struct TestResults {
    TriangleMesh out_mesh;

    double time_ms {};
    // TODO: add metrics
};

class TestMethod {
public:
    TestMethod() = delete;
    TestMethod(const std::string& name) {
        this->name = name;
    }

    std::string name;
    virtual TestResults run(Eigen::Ref<sdf::Vector> sdf_at_points, Eigen::Ref<Eigen::Vector3i> resolution, float spacing, AABB bounds) = 0;
};
