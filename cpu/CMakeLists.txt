cmake_minimum_required(VERSION 3.10)

project(SwinTrans)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
# onnx
include_directories(/home/nemo/engine/onnx_swintrans/onnx/include)
link_directories(/home/nemo/engine/onnx_swintrans/onnx/lib)

#opencv
include_directories(/home/nemo/software/environment/opencv-4.4/include/opencv4)
link_directories(/home/nemo/software/environment/opencv-4.4/lib)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

add_executable(SwinTrans ${PROJECT_SOURCE_DIR}/demo.cpp)
target_link_libraries(SwinTrans opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs opencv_dnn onnxruntime)

add_definitions(-Ofast -pthread)

