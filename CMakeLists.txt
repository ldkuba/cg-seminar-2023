project(SeminarCG)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_subdirectory(extern/pybind11)
add_subdirectory(extern/sdf)
add_subdirectory(extern/assimp)
add_subdirectory(extern/meshview)

file(GLOB_RECURSE SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.hpp)

add_executable(SeminarCG ${SOURCES})
target_link_libraries(SeminarCG PUBLIC sdf meshview assimp)
target_include_directories(SeminarCG 
    PUBLIC 
    extern/sdf 
    extern/assimp 
    extern/meshview 
    extern/pybind11 
    src)
