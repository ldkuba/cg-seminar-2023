#pragma once

// Fix M_PI not being defined on Windows
#define _USE_MATH_DEFINES

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include "sdf/sdf.hpp"

#include <assimp/Importer.hpp>  // C++ importer interface
#include <assimp/scene.h>       // Output data structure
#include <assimp/postprocess.h> // Post processing flags

#include "test_method.hpp"

class TestingPipeline {
public:
    struct PipelineSettings {
        int resolution;
    };

    TestingPipeline(const PipelineSettings& settings);
    ~TestingPipeline();

    void createTests();
    void run(const std::string& filename);
    std::unique_ptr<TestResults>& getResults(const std::string& method_name, int id = 0);

private:
    sdf::Points TestingPipeline::assimp_to_eigen_verts(aiVector3D* vertices, int num_vertices);
    sdf::Triangles TestingPipeline::assimp_to_eigen_indices(aiFace* indices, int num_indices);

    PipelineSettings _settings;

    std::vector<std::unique_ptr<TestMethod>> tests;
    std::unordered_map<std::string, std::vector<std::unique_ptr<TestResults>>> test_results;

    Assimp::Importer _importer;
};
