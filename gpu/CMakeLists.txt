cmake_minimum_required(VERSION 3.10)

project(SwinTransGpu)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)

#cuda 
include_directories(/usr/local/cuda-11.4/include)
link_directories(/usr/local/cuda-11.4/lib64)
# onnx
include_directories(/home/nemo/engine/onnx_swintrans/onnx-gpu/include)
link_directories(/home/nemo/engine/onnx_swintrans/onnx-gpu/lib)

#opencv
include_directories(/home/nemo/software/environment/opencv-4.4/include/opencv4)
link_directories(/home/nemo/software/environment/opencv-4.4/lib)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

add_executable(SwinTransGpu ${PROJECT_SOURCE_DIR}/demo.cpp)
target_link_libraries(SwinTransGpu opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs opencv_dnn onnxruntime)

add_definitions(-Ofast -pthread)

