#pragma once

#include "sdf/sdf.hpp"

struct TriangleMesh {
    sdf::Points vertices;
    sdf::Triangles indices;
};
