#include <iostream>

#include "testing_pipeline.hpp"
#include "methods/marching_cubes.hpp"

int main()
{
    TestingPipeline pipeline(TestingPipeline::PipelineSettings{200, 10000, 15.0f});
    pipeline.add_test_method(std::make_unique<MarchingCubes>());

    pipeline.run("models/lantern.obj", "test");

    TestEntry& entry = pipeline.getEntry("lantern.obj");
    std::cout << "Entry name: " << entry.name << ", chamfer dist: " << entry.results[0]->metrics.chamfer_distance << ", f1-score: " << entry.results[0]->metrics.f1_score;
    std::cout << ", edge chamfer dist: " << entry.results[0]->metrics.edge_chamfer_distance << ", edge f1-score: " << entry.results[0]->metrics.edge_f1_score;
    std::cout << ", inaccurate normals: " << entry.results[0]->metrics.inaccurate_normals << std::endl;

    return 0;
}
