#include "testing_pipeline.hpp"

int main()
{
    TestingPipeline pipeline(TestingPipeline::PipelineSettings{20});
    pipeline.run("models/lantern.obj");

    // meshview::Viewer viewer;
    // viewer.wireframe = true;
    // viewer.add_point_cloud(points, sdf_colors);
    // viewer.add_mesh(vertices, indices);
    // viewer.show();

    return 0;
}
