// Fix M_PI not being defined on Windows
#define _USE_MATH_DEFINES

#include <iostream>
#include <Eigen/Dense>
#include "sdf/sdf.hpp"
#include "meshview/meshview.hpp"

#include <assimp/Importer.hpp>  // C++ importer interface
#include <assimp/scene.h>       // Output data structure
#include <assimp/postprocess.h> // Post processing flags

sdf::Points convert_vertices(aiVector3D* vertices, int num_vertices)
{
    sdf::Points vertices_eigen(num_vertices, 3);
    for(int i = 0; i < num_vertices; i++) {
        vertices_eigen(i, 0) = vertices[i].x;
        vertices_eigen(i, 1) = vertices[i].y;
        vertices_eigen(i, 2) = vertices[i].z;
    }
    return vertices_eigen;
}

sdf::Triangles convert_indices(aiFace* indices, int num_indices)
{
    sdf::Triangles indices_eigen(num_indices, 3);
    for(int i = 0; i < num_indices; i++) {
        indices_eigen(i, 0) = indices[i].mIndices[0];
        indices_eigen(i, 1) = indices[i].mIndices[1];
        indices_eigen(i, 2) = indices[i].mIndices[2];
    }
    return indices_eigen;
}

int main()
{
    Assimp::Importer importer;
    const aiScene *scene = importer.ReadFile("models/lantern.obj",
                                                 aiProcess_Triangulate |
                                                 aiProcess_JoinIdenticalVertices);

    if(!scene) {
        std::cout << "Error loading scene" << std::endl;
        return 1;
    }
    std::cout << "Scene loaded" << std::endl;
    std::cout << "Number of meshes: " << scene->mNumMeshes << std::endl;

    aiMesh *mesh = scene->mMeshes[0];
    sdf::Points vertices = convert_vertices(mesh->mVertices, mesh->mNumVertices);
    sdf::Triangles indices = convert_indices(mesh->mFaces, mesh->mNumFaces);

    sdf::SDF sdf(vertices, indices);
    
    float size_x = sdf.aabb()(3) - sdf.aabb()(0);
    float size_y = sdf.aabb()(4) - sdf.aabb()(1);
    float size_z = sdf.aabb()(5) - sdf.aabb()(2);
    float diagonal = sqrt(size_x * size_x + size_y * size_y + size_z * size_z);

    const int resolution = 20;
    sdf::Points points(resolution * resolution * resolution, 3);
    for(int k = 0; k < resolution; k++) {
        for(int j = 0; j < resolution; j++) {
            for(int i = 0; i < resolution; i++) {
                points(i + j * resolution + k * resolution * resolution, 0) = (sdf.aabb()(0)) + (i / (float)resolution) * size_x;
                points(i + j * resolution + k * resolution * resolution, 1) = (sdf.aabb()(1)) + (j / (float)resolution) * size_y;
                points(i + j * resolution + k * resolution * resolution, 2) = (sdf.aabb()(2)) + (k / (float)resolution) * size_z;
            }
        }
    }

    sdf::Vector sdf_at_points = sdf(points);
    sdf::Points sdf_colors(resolution * resolution * resolution, 3);

    for(int i = 0; i < resolution * resolution * resolution; i++) {
        float sdf_value = sdf_at_points(i);
        if(sdf_value < 0.0f) {
            sdf_colors(i, 0) = 1.0f;
            sdf_colors(i, 1) = abs(sdf_value) * 1.8f;
            sdf_colors(i, 2) = 1.0f;
        } else {
            sdf_colors(i, 0) = abs(sdf_value) * 1.8f;
            sdf_colors(i, 1) = 1.0f;
            sdf_colors(i, 2) = 1.0f;
        }
    }

    std::cout << sdf_at_points.maxCoeff() << std::endl;
    std::cout << size_x << " " << size_y << " " << size_z << std::endl;

    meshview::Viewer viewer;
    viewer.wireframe = true;
    viewer.add_point_cloud(points, sdf_colors);
    viewer.add_mesh(vertices, indices);
    viewer.show();

    return 0;
}
