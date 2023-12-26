#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <cuda.h>
#include "util_cuda.hpp"

#define PI 3.14159265358979323846

CudaUtil::CudaUtil(int edge_set_num_points, int normal_accuracy_num_points) {
    // Allocate memory for edge set
    _edge_set_data.point_cloud_size = edge_set_num_points;
    cudaMalloc(&_edge_set_data.point_cloud, edge_set_num_points*sizeof(Eigen::Vector3f));
    cudaMalloc(&_edge_set_data.normals, edge_set_num_points*sizeof(Eigen::Vector3f));
    cudaMalloc(&_edge_set_data.edge_mask_cuda, edge_set_num_points*sizeof(char));
    _edge_set_data.edge_mask_host = new char[edge_set_num_points];

    // Allocate memory for normal accuracy
    _angles_data.normals_size = normal_accuracy_num_points;
    cudaMalloc(&_angles_data.out_points, normal_accuracy_num_points*sizeof(Eigen::Vector3f));
    cudaMalloc(&_angles_data.out_normals, normal_accuracy_num_points*sizeof(Eigen::Vector3f));
    cudaMalloc(&_angles_data.gt_points, normal_accuracy_num_points*sizeof(Eigen::Vector3f));
    cudaMalloc(&_angles_data.gt_normals, normal_accuracy_num_points*sizeof(Eigen::Vector3f));
    cudaMalloc(&_angles_data.angles_cuda, normal_accuracy_num_points*sizeof(float));
    _angles_data.angles_host = new float[normal_accuracy_num_points];
}

CudaUtil::~CudaUtil() {
    // Free memory for edge set
    cudaFree(_edge_set_data.point_cloud);
    cudaFree(_edge_set_data.normals);
    cudaFree(_edge_set_data.edge_mask_cuda);
    delete[] _edge_set_data.edge_mask_host;

    // Free memory for normal accuracy
    cudaFree(_angles_data.out_normals);
    cudaFree(_angles_data.gt_normals);
    cudaFree(_angles_data.angles_cuda);
    delete[] _angles_data.angles_host;
}

__global__ void cu_compute_edge_set(Eigen::Vector3f* point_cloud, Eigen::Vector3f* normals, char* out_mask, size_t num_points, float normal_epsilon, float distance_epsilon) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    Eigen::Vector3f p = point_cloud[idx];
    Eigen::Vector3f n = normals[idx];

    char is_edge = 0;
    for (int i = 0; i < num_points; i++) {
        if (i == idx) continue;
        Eigen::Vector3f q = point_cloud[i];
        Eigen::Vector3f r = normals[i];

        float dist = (q-p).norm();
        if (dist > distance_epsilon) continue;

        float dot = n.dot(r);
        if (abs(dot) > normal_epsilon) continue;

        is_edge = 1;
        break;
    }

    out_mask[idx] = is_edge;
}

std::vector<int> CudaUtil::compute_edge_set(const Points& point_cloud, const Points& normals, float normal_epsilon, float distance_epsilon) {

    size_t num_points = point_cloud.rows();

    // CUDA pointers
    Eigen::Vector3f* point_cloud_device;
    Eigen::Vector3f* normals_device;
    char* out_mask_device;
    
    // Host pointers
    char* out_mask;

    if(num_points != _edge_set_data.point_cloud_size) {
        cudaMalloc(&point_cloud_device, num_points*sizeof(Eigen::Vector3f));
        cudaMalloc(&normals_device, num_points*sizeof(Eigen::Vector3f));
        cudaMalloc(&out_mask_device, num_points*sizeof(char));
        out_mask = new char[num_points];
    } else {
        point_cloud_device = _edge_set_data.point_cloud;
        normals_device = _edge_set_data.normals;
        out_mask_device = _edge_set_data.edge_mask_cuda;
        out_mask = _edge_set_data.edge_mask_host;
    }

    cudaMemcpy(point_cloud_device, point_cloud.data(), num_points*sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice);
    cudaMemcpy(normals_device, normals.data(), num_points*sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice);

    // Run kernel
    cu_compute_edge_set<<<(num_points+1023)/1024,1024>>>(point_cloud_device, normals_device, out_mask_device, num_points, normal_epsilon, distance_epsilon);

    // Copy back
    cudaMemcpy(out_mask, out_mask_device, num_points*sizeof(char), cudaMemcpyDeviceToHost);

    std::vector<int> ret;
    for(size_t i = 0; i < num_points; i++) {
        if (out_mask[i] == 1) ret.push_back(i); 
    }

    // Free memory
    if(num_points != _edge_set_data.point_cloud_size) {
        cudaFree(point_cloud_device);
        cudaFree(normals_device);
        cudaFree(out_mask_device);
        delete[] out_mask;
    }

    return ret;
}

__global__ void cu_compute_angles(Eigen::Vector3f* out_points, Eigen::Vector3f* out_normals, Eigen::Vector3f* gt_points, Eigen::Vector3f* gt_normals, float* angles, size_t num_normals) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_normals) return;

    Eigen::Vector3f n = out_normals[idx];
    Eigen::Vector3f gt_n;
    
    // Find closest point
    float min_dist = -1.0;
    int min_idx = -1;
    for (int i = 0; i < num_normals; i++) {
        float dist = (out_points[idx] - gt_points[i]).norm();
        if (dist < min_dist || min_dist < 0.0) {
            min_dist = dist;
            min_idx = i;
        }
    }

    gt_n = gt_normals[min_idx];

    // Calculate angle
    float dot = n.dot(gt_n);
    float norm_dot = dot / (n.norm() * gt_n.norm());
    angles[idx] = acos(fmax(fmin(norm_dot, 1.0f), -1.0f)) * 180.0 / PI;
}

Eigen::VectorXf CudaUtil::compute_angles(const Points& out_points, const Points& out_normals, const Points& gt_points, const Points& gt_normals, float angle_threshold) {
    size_t num_normals = out_normals.rows();

    // CUDA pointers
    Eigen::Vector3f* out_points_device;
    Eigen::Vector3f* out_normals_device;
    Eigen::Vector3f* gt_points_device;
    Eigen::Vector3f* gt_normals_device;
    float* out_angle_device;

    // Host pointers
    float* out_angle;

    if(num_normals != _angles_data.normals_size) {
        cudaMalloc(&out_points_device, num_normals*sizeof(Eigen::Vector3f));
        cudaMalloc(&out_normals_device, num_normals*sizeof(Eigen::Vector3f));
        cudaMalloc(&gt_points_device, num_normals*sizeof(Eigen::Vector3f));
        cudaMalloc(&gt_normals_device, num_normals*sizeof(Eigen::Vector3f));
        cudaMalloc(&out_angle_device, num_normals*sizeof(float));
        out_angle = new float[num_normals];
    } else {
        out_points_device = _angles_data.out_points;
        out_normals_device = _angles_data.out_normals;
        gt_points_device = _angles_data.gt_points; 
        gt_normals_device = _angles_data.gt_normals;
        out_angle_device = _angles_data.angles_cuda;
        out_angle = _angles_data.angles_host;
    }

    cudaMemcpy(out_points_device, out_points.data(), num_normals*sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice);
    cudaMemcpy(out_normals_device, out_normals.data(), num_normals*sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice);
    cudaMemcpy(gt_points_device, gt_points.data(), num_normals*sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice);
    cudaMemcpy(gt_normals_device, gt_normals.data(), num_normals*sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice);

    // Run kernel
    cu_compute_angles<<<(num_normals+1023)/1024,1024>>>(out_points_device, out_normals_device, gt_points_device, gt_normals_device, out_angle_device, num_normals);

    // Copy back
    cudaMemcpy(out_angle, out_angle_device, num_normals*sizeof(float), cudaMemcpyDeviceToHost);

    Eigen::VectorXf ret(num_normals);
    for(size_t i = 0; i < num_normals; i++) {
        ret(i) = abs(out_angle[i]);
    }

    // Free memory
    if(num_normals != _angles_data.normals_size) {
        cudaFree(out_points_device);
        cudaFree(out_normals_device);
        cudaFree(gt_points_device);
        cudaFree(gt_normals_device);
        cudaFree(out_angle_device);
        delete[] out_angle;
    }

    return ret;
}

__global__ void cu_compute_normal_accuracy(Eigen::Vector3f* vertices, Eigen::Vector3i* indices, float* aspect_ratios, float* min_angles, size_t num_triangles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_triangles) return;

    Eigen::Vector3f v0 = vertices[indices[idx](0)];
    Eigen::Vector3f v1 = vertices[indices[idx](1)];
    Eigen::Vector3f v2 = vertices[indices[idx](2)];

    float a = (v1 - v0).norm();
    float b = (v2 - v0).norm();
    float c = (v2 - v1).norm();

    // Aspect ratio
    float s = (a + b + c) / 2.0f;
    aspect_ratios[idx] = (a * b * c) / (8.0f * (s-a) * (s-b) * (s-c));

    // Min angle
    float min_angle = 0.0f;
    if(a <= b && a <= c) {
        float cos_a = fmax(fmin((b*b + c*c - a*a) / (2.0f * b * c), 1.0f), -1.0f);
        min_angle = acos(cos_a) * 180.0 / PI;
    } else if(b <= a && b <= c) {
        float cos_b = fmax(fmin((a*a + c*c - b*b) / (2.0f * a * c), 1.0f), -1.0f);
        min_angle = acos(cos_b) * 180.0 / PI;
    } else {
        float cos_c = fmax(fmin((b*b + a*a - c*c) / (2.0f * b * a), 1.0f), -1.0f);
        min_angle = acos(cos_c) * 180.0 / PI;
    }
    min_angles[idx] = min_angle;
}

std::pair<Eigen::VectorXf, Eigen::VectorXf> CudaUtil::compute_aspect_ratios_and_min_angles(const TriangleMesh& mesh) {
    size_t num_triangles = mesh.indices.rows();

    // CUDA pointers
    Eigen::Vector3f* vertices_device;
    Eigen::Vector3i* indices_device;
    float* out_aspect_ratios_device;
    float* out_min_angles_device;

    // Host pointers
    float* out_aspect_ratios;
    float* out_min_angles;

    cudaMalloc(&vertices_device, mesh.vertices.rows()*sizeof(Eigen::Vector3f));
    cudaMalloc(&indices_device, num_triangles*sizeof(Eigen::Vector3i));
    cudaMalloc(&out_aspect_ratios_device, num_triangles*sizeof(float));
    cudaMalloc(&out_min_angles_device, num_triangles*sizeof(float));
    out_aspect_ratios = new float[num_triangles];
    out_min_angles = new float[num_triangles];

    cudaMemcpy(vertices_device, mesh.vertices.data(), mesh.vertices.rows()*sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice);
    cudaMemcpy(indices_device, mesh.indices.data(), num_triangles*sizeof(Eigen::Vector3i), cudaMemcpyHostToDevice);

    // Run kernel
    cu_compute_normal_accuracy<<<(num_triangles+1023)/1024,1024>>>(vertices_device, indices_device, out_aspect_ratios_device, out_min_angles_device, num_triangles);

    // Copy back
    cudaMemcpy(out_aspect_ratios, out_aspect_ratios_device, num_triangles*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_min_angles, out_min_angles_device, num_triangles*sizeof(float), cudaMemcpyDeviceToHost);

    Eigen::VectorXf aspect_ratios(num_triangles);
    Eigen::VectorXf min_angles(num_triangles);
    for(size_t i = 0; i < num_triangles; i++) {
        aspect_ratios(i) = out_aspect_ratios[i];
        min_angles(i) = out_min_angles[i];
    }

    // Free memory
    cudaFree(vertices_device);
    cudaFree(indices_device);
    cudaFree(out_aspect_ratios_device);
    cudaFree(out_min_angles_device);
    delete[] out_aspect_ratios;
    delete[] out_min_angles;

    return std::make_pair(aspect_ratios, min_angles);
}
