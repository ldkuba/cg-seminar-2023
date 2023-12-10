#pragma once

// Fix M_PI not being defined on Windows
#define _USE_MATH_DEFINES

#include <memory>
#include <string>
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

private:
    sdf::Points TestingPipeline::assimp_to_eigen_verts(aiVector3D* vertices, int num_vertices);
    sdf::Triangles TestingPipeline::assimp_to_eigen_indices(aiFace* indices, int num_indices);

    PipelineSettings _settings;

    std::vector<std::unique_ptr<TestMethod>> tests;
    Assimp::Importer _importer;
};
