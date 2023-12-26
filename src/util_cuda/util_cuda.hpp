#pragma once
#include <vector>
#include <Eigen/Dense>
#include "triangle_mesh.hpp"

using Points = Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>;

class CudaUtil {
public:
    CudaUtil(int edge_set_num_points, int normal_accuracy_num_points);
    ~CudaUtil();

    std::vector<int> compute_edge_set(const Points& point_cloud, const Points& normals, float normal_epsilon, float distance_epsilon);
    Eigen::VectorXf compute_angles(const Points& out_points, const Points& out_normals, const Points& gt_points, const Points& gt_normals, float angle_threshold);
    std::pair<Eigen::VectorXf, Eigen::VectorXf> compute_aspect_ratios_and_min_angles(const TriangleMesh& mesh);

private:
    // Edge set
    struct EdgeSetData {
        int point_cloud_size;
        Eigen::Vector3f* point_cloud;
        Eigen::Vector3f* normals;
        char* edge_mask_cuda;
        char* edge_mask_host;
    };
    EdgeSetData _edge_set_data;

    // Normal accuracy
    struct AnglesData {
        int normals_size;
        Eigen::Vector3f* out_points;
        Eigen::Vector3f* out_normals;
        Eigen::Vector3f* gt_points;
        Eigen::Vector3f* gt_normals;
        float* angles_cuda;
        float* angles_host;
    };
    AnglesData _angles_data;
};
