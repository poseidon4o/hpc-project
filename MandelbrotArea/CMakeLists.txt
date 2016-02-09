cmake_minimum_required(VERSION 2.8)

project(MandelbrotArea)

file(GLOB HEADERS *.h)
file(GLOB SOURCES *.cpp)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

if(WIN32)
else()
	add_definitions(-std=c++11 -mavx -O3)
endif()

