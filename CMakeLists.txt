project(ncnnyoloface)

cmake_minimum_required(VERSION 3.10)

find_package(OpenCV REQUIRED)

set(ncnn_DIR "~/library/ncnn/build/install/lib/cmake/ncnn" CACHE PATH "Directory that contains ncnnConfig.cmake")
find_package(ncnn REQUIRED)

# include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(ncnnyoloface main.cpp yoloface.cpp)

# add_library(ncnnyoloface SHARED yolofacencnn.cpp yoloface.cpp ndkcamera.cpp)

target_link_libraries(ncnnyoloface ncnn ${OpenCV_LIBS})
