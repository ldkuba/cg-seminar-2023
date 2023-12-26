#include <iostream>

#include "testing_pipeline.hpp"
#include "methods/marching_cubes.hpp"
#include "methods/reach_for_the_spheres.hpp"

int main()
{
    TestingPipeline pipeline(TestingPipeline::PipelineSettings{50, 10000, 15.0f, 10.0f});
    //pipeline.add_test_method(std::make_unique<ReachForTheSpheres>());
    pipeline.add_test_method(std::make_unique<MarchingCubes>());

    pipeline.run("models/lantern.obj", "test");

    TestEntry& entry = pipeline.getEntry("lantern.obj");
    std::cout << "Entry name: " << entry.name << ", chamfer dist: " << entry.results[0]->metrics.chamfer_distance << ", f1-score: " << entry.results[0]->metrics.f1_score;
    std::cout << ", edge chamfer dist: " << entry.results[0]->metrics.edge_chamfer_distance << ", edge f1-score: " << entry.results[0]->metrics.edge_f1_score;
    std::cout << ", inaccurate normals: " << entry.results[0]->metrics.inaccurate_normals << ", mean normal error: " << entry.results[0]->metrics.mean_normal_error;
    std::cout << ", mean aspect ratio: " << entry.results[0]->metrics.mean_aspect_ratio << ", min aspect ratio: " << entry.results[0]->metrics.min_aspect_ratio;
    std::cout << ", max aspect ratio: " << entry.results[0]->metrics.max_aspect_ratio << ", percent sliver triangles: " << entry.results[0]->metrics.percent_sliver_triangles << std::endl;

    return 0;
}
