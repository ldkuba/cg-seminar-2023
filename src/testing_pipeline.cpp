#include "testing_pipeline.hpp"

#include <cmath>
#include <iostream>

#include "pybind11/pybind11.h"
#include "pybind11/embed.h"

#include "meshview/meshview.hpp"

#include "methods/marching_cubes.hpp"

namespace py = pybind11;

TestingPipeline::TestingPipeline(const PipelineSettings& settings):
    _settings(settings) {
    py::initialize_interpreter();
    auto main_module = py::module_::import("__main__");

    // Register results struct
    py::class_<TriangleMesh>(main_module, "TriangleMesh")
        .def(py::init<>())
        .def_readwrite("vertices", &TriangleMesh::vertices)
        .def_readwrite("indices", &TriangleMesh::indices);

    py::class_<TestResults>(main_module, "TestResults")
        .def(py::init<>())
        .def_readwrite("out_mesh", &TestResults::out_mesh)
        .def_readwrite("time_ms", &TestResults::time_ms);

    createTests();
}

void TestingPipeline::createTests() {
    tests.push_back(std::make_unique<MarchingCubes>());
}

sdf::Points TestingPipeline::assimp_to_eigen_verts(aiVector3D* vertices, int num_vertices)
{
    sdf::Points vertices_eigen(num_vertices, 3);
    for(int i = 0; i < num_vertices; i++) {
        vertices_eigen(i, 0) = vertices[i].x;
        vertices_eigen(i, 1) = vertices[i].y;
        vertices_eigen(i, 2) = vertices[i].z;
    }
    return vertices_eigen;
}

sdf::Triangles TestingPipeline::assimp_to_eigen_indices(aiFace* indices, int num_indices)
{
    sdf::Triangles indices_eigen(num_indices, 3);
    for(int i = 0; i < num_indices; i++) {
        indices_eigen(i, 0) = indices[i].mIndices[0];
        indices_eigen(i, 1) = indices[i].mIndices[1];
        indices_eigen(i, 2) = indices[i].mIndices[2];
    }
    return indices_eigen;
}

std::unique_ptr<TestResults>& TestingPipeline::getResults(const std::string& method_name, int id) {
    return test_results[method_name].at(id);
}

void TestingPipeline::run(const std::string& filename) {

    // Import model
    const aiScene *scene = _importer.ReadFile(filename,
                                                 aiProcess_Triangulate |
                                                 aiProcess_JoinIdenticalVertices);

    if(!scene) {
        std::cout << "Error loading scene" << std::endl;
        return;
    }
    std::cout << "Scene loaded" << std::endl;
    std::cout << "Number of meshes: " << scene->mNumMeshes << std::endl;

    aiMesh *mesh = scene->mMeshes[0];
    sdf::Points vertices = assimp_to_eigen_verts(mesh->mVertices, mesh->mNumVertices);
    sdf::Triangles indices = assimp_to_eigen_indices(mesh->mFaces, mesh->mNumFaces);

    // Create SDF
    sdf::SDF sdf(vertices, indices);

    // Create volume sample points
    const AABB aabb = sdf.aabb();

    float size_x = aabb(3) - aabb(0);
    float size_y = aabb(4) - aabb(1);
    float size_z = aabb(5) - aabb(2);
    float max_size = std::max(std::max(size_x, size_y), size_z);
    float min_bounds = std::min(std::min(aabb(0), aabb(1)), aabb(2));
    float diagonal = sqrt(3.0) * max_size;

    float spacing = max_size / static_cast<float>(_settings.resolution - 1);
    Eigen::Vector3i resolution = Eigen::Vector3i(static_cast<int>(std::ceil(size_x / spacing)) + 1, static_cast<int>(std::ceil(size_y / spacing)) + 1, static_cast<int>(std::ceil(size_z / spacing)) + 1);

    sdf::Points points(resolution(0) * resolution(1) * resolution(2), 3);
    for(int k = 0; k < resolution(2); k++) {
        for(int j = 0; j < resolution(1); j++) {
            for(int i = 0; i < resolution(0); i++) {
                float x_val = aabb(0) + i * spacing;
                float y_val = aabb(1) + j * spacing;
                float z_val = aabb(2) + k * spacing;
                points(i + j * resolution(0) + k * resolution(1) * resolution(0), 0) = x_val;
                points(i + j * resolution(0) + k * resolution(1) * resolution(0), 1) = y_val;
                points(i + j * resolution(0) + k * resolution(1) * resolution(0), 2) = z_val;
            }
        }
    }

    // Calculate sdf values at sample points
    sdf::Vector sdf_at_points = sdf(points);

    for (auto& test : tests) {
        std::unique_ptr<TestResults> results = test->run(sdf_at_points, resolution, spacing, aabb);
        std::cout << test->name << ": " << results->time_ms << "ms" << std::endl;
        // // Visualize sample points and mesh
        // meshview::Viewer viewer;
        // viewer.wireframe = true;
        // viewer.add_point_cloud(points);
        // viewer.add_mesh(results->out_mesh.vertices, results->out_mesh.indices);
        // viewer.show();

        test_results[test->name].push_back(std::move(results));
    }
}

TestingPipeline::~TestingPipeline() {
    py::finalize_interpreter();
}
