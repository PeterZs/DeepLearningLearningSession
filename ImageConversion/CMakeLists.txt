cmake_minimum_required(VERSION 3.0)

project(ImageConversion)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#set(BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/build")
set(CMAKE_BUILD_TYPE Release)

find_package(OpenCV REQUIRED)
#find_package(GLOG REQUIRED)
find_package(gflags REQUIRED)

#file(GLOB SOURCES "*.cpp")
add_executable(ConstantLabel ConstantLabel.cpp )
target_link_libraries(ConstantLabel ${OpenCV_LIBS} ${GFlags_LIBS} gflags)

add_executable(CifarImageExtraction CifarImageExtraction.cpp )
target_link_libraries(CifarImageExtraction ${OpenCV_LIBS} ${GFlags_LIBS} gflags)

add_executable(Differentiator Differentiator.cpp )
target_link_libraries(Differentiator ${OpenCV_LIBS} ${GFlags_LIBS} gflags)

add_executable(WriteLists WriteLists.cpp )
target_link_libraries(WriteLists ${OpenCV_LIBS} ${GFlags_LIBS} gflags)

add_executable(WritePairs WritePairs.cpp )
target_link_libraries(WritePairs ${OpenCV_LIBS} ${GFlags_LIBS} gflags)

#add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../base)
