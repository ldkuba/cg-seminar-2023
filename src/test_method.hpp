#pragma once

#include <string>
#include <memory>
#include <unordered_set>
#include <utility>

#include <Eigen/Dense>
#include "sdf/sdf.hpp"

#include "triangle_mesh.hpp"

using AABB = Eigen::Ref<const Eigen::Matrix<float, 6, 1>>;

struct EvaluationMetrics {
    float chamfer_distance;
    float f1_score;
    float edge_chamfer_distance;
    float edge_f1_score;
    float inaccurate_normals;
    float mean_normal_error;
    float mean_aspect_ratio;
    float min_aspect_ratio;
    float max_aspect_ratio;
};

// Represents the results of one method on one model/dataset element
struct TestResults {
    std::string method;

    // === Extracted mesh ===
    TriangleMesh out_mesh;

    // === Metrics ===
    double time_ms {};
    EvaluationMetrics metrics;
};

// Represents one model/dataset element in the testing piepline
struct TestEntry {
    std::string name;
    std::vector<std::unique_ptr<TestResults>> results;

    // Store point cloud sampled from input mesh for evaluation metrics to avoid recomputing
    std::pair<sdf::Points, sdf::Points> in_mesh_point_cloud;
    std::pair<sdf::Points, sdf::Points> in_mesh_point_cloud_edge;

    // Dataset categories that this mesh belongs to
    std::unordered_set<std::string> categories;
};

// Used to pass input data to the methods
struct TestInputData {
    Eigen::Ref<sdf::Points> sdf_points;
    Eigen::Ref<sdf::Vector> sdf_at_points;
    Eigen::Ref<Eigen::Vector3i> resolution;
    float spacing;
    AABB bounds;
};

class TestMethod {
public:
    TestMethod() = delete;
    TestMethod(const std::string& name) {
        this->name = name;
    }

    std::string name;
    virtual std::unique_ptr<TestResults> run(const TestInputData& input) = 0;
};
