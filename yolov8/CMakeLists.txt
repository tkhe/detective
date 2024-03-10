cmake_minimum_required(VERSION 3.10)

project(yolov8)

cmake_minimum_required(VERSION 3.10)

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/../opencv-mobile/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/../ncnn/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

add_executable(yolov8 src/yolov8.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} ncnn)
