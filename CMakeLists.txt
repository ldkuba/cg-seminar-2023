project(SeminarCG)
set(CMAKE_CXX_STANDARD 17)

include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
    message(STATUS "No CUDA support")
else()
    enable_language(CUDA)
    message(STATUS "CUDA support enabled")
    add_compile_definitions(HAS_CUDA)
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_subdirectory(extern/pybind11)
add_subdirectory(extern/sdf)
add_subdirectory(extern/assimp)
add_subdirectory(extern/meshview)

file(GLOB_RECURSE SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.hpp)

if(CMAKE_CUDA_COMPILER)
    file(GLOB_RECURSE CUDA_SOURCES ${PROJECT_SOURCE_DIR}/src/util_cuda/*)
    list(APPEND SOURCES ${CUDA_SOURCES})
endif()

add_executable(SeminarCG ${SOURCES})
target_link_libraries(SeminarCG PUBLIC sdf meshview assimp pybind11::embed)
target_include_directories(SeminarCG 
    PUBLIC 
    extern/sdf 
    extern/assimp 
    extern/meshview 
    extern/pybind11/include
    src)

if(CMAKE_CUDA_COMPILER)
    set_property(TARGET SeminarCG PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endif()
