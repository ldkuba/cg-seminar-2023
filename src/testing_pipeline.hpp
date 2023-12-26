#pragma once

// Fix M_PI not being defined on Windows
#define _USE_MATH_DEFINES

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "sdf/sdf.hpp"

#include <assimp/Importer.hpp>  // C++ importer interface
#include <assimp/scene.h>       // Output data structure
#include <assimp/postprocess.h> // Post processing flags

#include "test_method.hpp"

#ifdef HAS_CUDA
#include "util_cuda/util_cuda.hpp"
#endif

class TestingPipeline {
public:
    struct PipelineSettings {
        unsigned int resolution;
        unsigned int evaluation_pt_size;

        float inacurate_normals_threshold;
        float sliver_triangle_angle;
    };

    TestingPipeline(const PipelineSettings& settings);
    ~TestingPipeline();

    void add_test_method(std::unique_ptr<TestMethod>&& method);
    void run(const std::string& filename, const std::string& category);
    void run(const std::vector<std::string>& filenames, const std::string& category);

    TestEntry& getEntry(const std::string& filename);

private:
    sdf::Points TestingPipeline::assimp_to_eigen_verts(aiVector3D* vertices, int num_vertices);
    sdf::Triangles TestingPipeline::assimp_to_eigen_indices(aiFace* indices, int num_indices);

    void evaluation_metrics(std::unique_ptr<TestResults>& results, const std::pair<sdf::Points,sdf::Points>& gt_point_cloud, const std::pair<sdf::Points,sdf::Points>& gt_point_cloud_edge);

    PipelineSettings _settings;

#ifdef HAS_CUDA
    CudaUtil _cuda_util;
#endif

    std::vector<std::unique_ptr<TestMethod>> tests;
    std::unordered_map<std::string, TestEntry> test_entries;

    Assimp::Importer _importer;
};
