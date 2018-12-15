cmake_minimum_required(VERSION 2.8)

#set( CMAKE_CXX_FLAGS "-std=c++11")

project(objectdetect)

set(CMAKE_MODULE_PATH "$ENV{3rdPartyLibs_DIR}")
# OpenCV lib
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


set(CMAKE_MFC_FLAG 1)
add_definitions("/D_CONSOLE /DUNICODE /D_UNICODE")   # set for both debug and release
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Gm /RTC1 /MDd /ZI /TP")    # set only for debug mode
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Oi /GL /MT /Gy /Zi /FS")   # set only for release mode

file (GLOB CPP_FILES src/*.cpp src/*.cc) 
file (GLOB H_FILES src/*.h) 
set (SOURCE_FILES ${CPP_FILES} ${H_FILES})

add_executable(objectdetect ${SOURCE_FILES})
target_link_libraries(objectdetect ${OpenCV_LIBS})



