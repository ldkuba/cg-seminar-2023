#include "testing_pipeline.hpp"

#include <cmath>
#include <iostream>
#include <chrono>

#include "pybind11/pybind11.h"
#include "pybind11/embed.h"
#include "pybind11/eigen.h"

#include "meshview/meshview.hpp"

#include "methods/marching_cubes.hpp"

#include "py_util.hpp"

namespace py = pybind11;

TestingPipeline::TestingPipeline(const PipelineSettings& settings):
    _settings(settings)
#ifdef HAS_CUDA
    ,_cuda_util(settings.evaluation_pt_size, settings.evaluation_pt_size)
#endif
{
    py::initialize_interpreter();
    auto main_module = py::module_::import("__main__");

    // Register results struct
    py::class_<TriangleMesh>(main_module, "TriangleMesh")
        .def(py::init<>())
        .def_readwrite("vertices", &TriangleMesh::vertices)
        .def_readwrite("indices", &TriangleMesh::indices);

    py::class_<EvaluationMetrics>(main_module, "EvaluationMetrics")
        .def(py::init<>())
        .def_readwrite("chamfer_distance", &EvaluationMetrics::chamfer_distance)
        .def_readwrite("f1_score", &EvaluationMetrics::f1_score)
        .def_readwrite("edge_chamfer_distance", &EvaluationMetrics::edge_chamfer_distance)
        .def_readwrite("edge_f1_score", &EvaluationMetrics::edge_f1_score);

    py::class_<TestResults>(main_module, "TestResults")
        .def(py::init<>())
        .def_readwrite("method", &TestResults::method)
        .def_readwrite("out_mesh", &TestResults::out_mesh)
        .def_readwrite("time_ms", &TestResults::time_ms)
        .def_readwrite("metrics", &TestResults::metrics);
}

void TestingPipeline::add_test_method(std::unique_ptr<TestMethod>&& method) {
    tests.push_back(std::move(method));
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

TestEntry& TestingPipeline::getEntry(const std::string& filename) {
    return test_entries[filename];
}

void TestingPipeline::run(const std::vector<std::string>& filenames, const std::string& category) {
    for (const auto& filename : filenames) {
        run(filename, category);
    }
}

void TestingPipeline::run(const std::string& filename, const std::string& category) {

    // Extract filename from path
    std::string model_name = filename.substr(filename.find_last_of("/\\") + 1);

    // Check if this model has already been processed
    auto entry_it = test_entries.find(model_name);
    if (entry_it != test_entries.end()) {
        auto& test_entry = entry_it->second;
        test_entry.categories.insert(category);
        return;
    }

    // Create new entry
    test_entries[model_name] = TestEntry{model_name, {}, {}, {}, {category}};
    auto& test_entry = test_entries[model_name];

    // Start running timer
    auto start_time = std::chrono::high_resolution_clock::now();

    // Import model
    // const aiScene *scene = _importer.ReadFile(filename,
    //                                              aiProcess_Triangulate |
    //                                              aiProcess_JoinIdenticalVertices);

    // if(!scene) {
    //     std::cout << "Error loading scene" << std::endl;
    //     return;
    // }
    // std::cout << "Scene loaded" << std::endl;
    // std::cout << "Number of meshes: " << scene->mNumMeshes << std::endl;

    // aiMesh *mesh = scene->mMeshes[0];
    //sdf::Points vertices = assimp_to_eigen_verts(mesh->mVertices, mesh->mNumVertices);
    //sdf::Triangles indices = assimp_to_eigen_indices(mesh->mFaces, mesh->mNumFaces);

    // Use trimesh in python for now because assimp is generating duplicate vertices if uvs or normals differ
    sdf::Points vertices;
    sdf::Triangles indices;
    PY_CALL_START
    auto locals = py::dict();
    locals["filename"] = filename;
    py::eval_file("src/load_mesh.py", py::globals(), locals);
    vertices = locals["vertices"].cast<sdf::Points>();
    indices = locals["indices"].cast<sdf::Triangles>();
    PY_CALL_END

    auto model_load_time = std::chrono::high_resolution_clock::now();
    std::cout << "Model loaded - " << std::chrono::duration_cast<std::chrono::milliseconds>(model_load_time - start_time).count() << "ms" << std::endl;
    std::cout << "Number of vertices: " << vertices.rows() << std::endl;

    // Create SDF
    sdf::SDF sdf(vertices, indices);

    // Create volume sample points
    AABB original_aabb = sdf.aabb();
    Eigen::Matrix<float, 6, 1> aabb = original_aabb;

    float size_x = aabb(3) - aabb(0);
    float size_y = aabb(4) - aabb(1);
    float size_z = aabb(5) - aabb(2);
    float max_size = std::max(std::max(size_x, size_y), size_z);
    
    // Expand aabb by 10%
    aabb(0) -= max_size * 0.1f;
    aabb(1) -= max_size * 0.1f;
    aabb(2) -= max_size * 0.1f;
    aabb(3) += max_size * 0.1f;
    aabb(4) += max_size * 0.1f;
    aabb(5) += max_size * 0.1f;

    size_x = aabb(3) - aabb(0);
    size_y = aabb(4) - aabb(1);
    size_z = aabb(5) - aabb(2);
        
    max_size = std::max(std::max(size_x, size_y), size_z);
    float min_bounds = std::min(std::min(aabb(0), aabb(1)), aabb(2));
    float diagonal = sqrt(3.0) * max_size;

    float spacing = max_size / static_cast<float>(_settings.resolution - 1);
    Eigen::Vector3i resolution = Eigen::Vector3i(static_cast<int>(std::ceil(size_x / spacing)) + 1, static_cast<int>(std::ceil(size_y / spacing)) + 1, static_cast<int>(std::ceil(size_z / spacing)) + 1);

    sdf::Points points(resolution(0) * resolution(1) * resolution(2), 3);
    for(int i = 0; i < resolution(0); i++) {
        for(int j = 0; j < resolution(1); j++) {
            for(int k = 0; k < resolution(2); k++) {
                float x_val = aabb(0) + i * spacing;
                float y_val = aabb(1) + j * spacing;
                float z_val = aabb(2) + k * spacing;
                points(k + j * resolution(2) + i * resolution(1) * resolution(2), 0) = x_val;
                points(k + j * resolution(2) + i * resolution(1) * resolution(2), 1) = y_val;
                points(k + j * resolution(2) + i * resolution(1) * resolution(2), 2) = z_val;
            }
        }
    }

    // Calculate sdf values at sample points
    sdf::Vector sdf_at_points = sdf(points);
    auto sdf_calc_time = std::chrono::high_resolution_clock::now();
    std::cout << "Calculated sdf values - " << std::chrono::duration_cast<std::chrono::milliseconds>(sdf_calc_time - model_load_time).count() << "ms" << std::endl;

    // Compute and cache ground truth point cloud
    test_entry.in_mesh_point_cloud = sdf.sample_surface(_settings.evaluation_pt_size);
    auto point_cloud_sample_time = std::chrono::high_resolution_clock::now();
    std::cout << "Sampled original surface - " << std::chrono::duration_cast<std::chrono::milliseconds>(point_cloud_sample_time - sdf_calc_time).count() << "ms" << std::endl;

    // Compute and cache ground truth point cloud edge as a subset of the point cloud
#ifdef HAS_CUDA
    std::vector<int> edge_points = _cuda_util.compute_edge_set(test_entry.in_mesh_point_cloud.first, test_entry.in_mesh_point_cloud.second, 0.2f, 0.1f);
#else
    std::vector<int> edge_points;
    throw std::runtime_error("CPU edge set computation not implemented yet!");
#endif
    auto edge_set_time = std::chrono::high_resolution_clock::now();
    std::cout << "Computed edge set - " << std::chrono::duration_cast<std::chrono::milliseconds>(edge_set_time - point_cloud_sample_time).count() << "ms" << std::endl;
    
    test_entry.in_mesh_point_cloud_edge.first = test_entry.in_mesh_point_cloud.first(edge_points, {0, 1, 2});
    test_entry.in_mesh_point_cloud_edge.second = test_entry.in_mesh_point_cloud.second(edge_points, {0, 1, 2});

    for (auto& test : tests) {
        std::unique_ptr<TestResults> results = test->run({points, sdf_at_points, resolution, spacing, aabb});
        std::cout << test->name << ": " << results->time_ms << "ms" << std::endl;

        // Visualize mesh
        sdf::Points pt_colors(points.rows(), 3);
        for (int i = 0; i < points.rows(); i++) {
            pt_colors(i, 0) = sdf_at_points(i) < 0.0f ? 1.0f : 0.0f;
            pt_colors(i, 1) = sdf_at_points(i) > 0.0f ? 1.0f : 0.0f;
            pt_colors(i, 2) = 0.0f;
        }

        meshview::Viewer viewer;
        viewer.wireframe = true;
        //viewer.add_mesh(results->out_mesh.vertices, results->out_mesh.indices);
        viewer.add_point_cloud(points, pt_colors);
        viewer.add_mesh(vertices, indices);
        viewer.show();

        meshview::Viewer viewer2;
        viewer2.wireframe = true;
        viewer2.add_mesh(results->out_mesh.vertices, results->out_mesh.indices);
        viewer2.add_point_cloud(points, pt_colors);
        //viewer2.add_mesh(vertices, indices);
        viewer2.show();

        // Run evaluation metrics
        evaluation_metrics(results, test_entry.in_mesh_point_cloud, test_entry.in_mesh_point_cloud_edge);

        test_entry.results.push_back(std::move(results));
    }
}

void TestingPipeline::evaluation_metrics(std::unique_ptr<TestResults>& results, const std::pair<sdf::Points,sdf::Points>& gt_point_cloud, const std::pair<sdf::Points,sdf::Points>& gt_point_cloud_edge) {
    // Prepare point cloud sampled from output mesh
    sdf::SDF point_cloud_sampler(results->out_mesh.vertices, results->out_mesh.indices);
    std::pair<sdf::Points, sdf::Points> point_cloud_pair = point_cloud_sampler.sample_surface(_settings.evaluation_pt_size);
    
    // compute edge set of output point cloud
#ifdef HAS_CUDA
    std::vector<int> edge_points = _cuda_util.compute_edge_set(point_cloud_pair.first, point_cloud_pair.second, 0.2f, 0.1f);
#else
    std::vector<int> edge_points;
    throw std::runtime_error("CPU edge set computation not implemented yet!");
#endif
    std::pair<sdf::Points, sdf::Points> point_cloud_pair_edge = {point_cloud_pair.first(edge_points, {0, 1, 2}), point_cloud_pair.second(edge_points, {0, 1, 2})};

    // Chamfer distance and f-score for point cloud and edge point cloud
    PY_CALL_START
    auto locals = py::dict();
    locals["out_point_cloud"] = point_cloud_pair.first;
    locals["out_point_cloud_edge"] = point_cloud_pair_edge.first;

    locals["gt_point_cloud"] = gt_point_cloud.first;
    locals["gt_point_cloud_edge"] = gt_point_cloud_edge.first;

    locals["eval_metrics"] = results->metrics;

    py::eval_file("src/evaluate.py", py::globals(), locals);
    results->metrics = locals["eval_metrics"].cast<EvaluationMetrics>();
    PY_CALL_END

    // Inaccurate normals < 5deg
#ifdef HAS_CUDA
    Eigen::VectorXf angles = _cuda_util.compute_angles(point_cloud_pair.first, point_cloud_pair.second, gt_point_cloud.first, gt_point_cloud.second, _settings.inacurate_normals_threshold);
#else
    Eigen::VectorXf angles;
    throw std::runtime_error("CPU normal angle computation not implemented yet!");
#endif
    results->metrics.inaccurate_normals = (angles.array() > _settings.inacurate_normals_threshold).count() / static_cast<float>(angles.size());
    results->metrics.mean_normal_error = angles.sum() / static_cast<float>(angles.size());

    // // Visualize normals
    // sdf::Points angle_colors(angles.size(), 3);
    // for (int i = 0; i < angles.size(); i++) {
    //     // angle_colors(i, 0) = angles(i) / 180.0f;
    //     // angle_colors(i, 1) = (180.0f - angles(i)) / 180.0f;
    //     // angle_colors(i, 2) = 0.0f;
    //     angle_colors(i, 0) = angles(i) > _settings.inacurate_normals_threshold ? 1.0f : 0.0f;
    //     angle_colors(i, 1) = angles(i) < _settings.inacurate_normals_threshold ? 1.0f : 0.0f;
    //     angle_colors(i, 2) = 0.0f;
    // }

    // meshview::Viewer viewer;
    // viewer.wireframe = true;
    // viewer.add_point_cloud(point_cloud_pair.first, angle_colors);
    // Eigen::Vector3f blue(0.0f, 0.0f, 1.0f);
    // for(int i = 0 < 0; i < point_cloud_pair.second.rows(); i++) {
    //     viewer.add_line(point_cloud_pair.first.row(i), point_cloud_pair.first.row(i) + point_cloud_pair.second.row(i) * 0.1f, angle_colors.row(i));
    //     //viewer.add_line(gt_point_cloud.first.row(i), gt_point_cloud.first.row(i) + gt_point_cloud.second.row(i) * 0.1f, blue);
    // }
    // viewer.add_mesh(results->out_mesh.vertices, results->out_mesh.indices);
    // viewer.show();

    // Triangle aspect ratio and % of sliver triangles
#ifdef HAS_CUDA
    std::pair<Eigen::VectorXf, Eigen::VectorXf> aspect_ratios_and_min_angles = _cuda_util.compute_aspect_ratios_and_min_angles(results->out_mesh);
    Eigen::VectorXf& aspect_ratios = aspect_ratios_and_min_angles.first;
    Eigen::VectorXf& min_angles = aspect_ratios_and_min_angles.second;
#else
    Eigen::VectorXf aspect_ratios;
    throw std::runtime_error("CPU aspect ratio computation not implemented yet!");
#endif
    results->metrics.mean_aspect_ratio = aspect_ratios.sum() / static_cast<float>(aspect_ratios.size());
    results->metrics.min_aspect_ratio = aspect_ratios.minCoeff();
    results->metrics.max_aspect_ratio = aspect_ratios.maxCoeff();

    results->metrics.percent_sliver_triangles = (min_angles.array() < _settings.sliver_triangle_angle).count() / static_cast<float>(min_angles.size());
}

TestingPipeline::~TestingPipeline() {
    py::finalize_interpreter();
}
