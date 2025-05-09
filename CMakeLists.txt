cmake_minimum_required(VERSION 3.30)
project(Parallel_Background_Subtraction)

set(CMAKE_CXX_STANDARD 20)

# Set OpenCV path (if needed on Windows)
set(ENV{OPENCV_DIR} "C:\\tools\\opencv\\build")

# Find required packages
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

# Include OpenCV and MPI headers
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${MPI_INCLUDE_PATH})

# 1. Sequential Median
add_executable(Squential_Module Squential_Module.cpp)
target_link_libraries(Squential_Module ${OpenCV_LIBS})

# 2. OpenMP Median
add_executable(OpenMP_Module OpenMP_Module.cpp)
target_link_libraries(OpenMP_Module ${OpenCV_LIBS} OpenMP::OpenMP_CXX)

# 3. MPI Median
add_executable(MPI_Module MPI_Module.cpp)
target_link_libraries(MPI_Module ${OpenCV_LIBS} MPI::MPI_CXX)

# 4. Full Demo or Combined Program
add_executable(Parallel_Background_Subtraction
        main.cpp
        Failure_Trials/Sequentail_Mean.cpp
        Failure_Trials/Sequential_Base.cpp
)
target_link_libraries(Parallel_Background_Subtraction
    ${OpenCV_LIBS}
    OpenMP::OpenMP_CXX
    MPI::MPI_CXX
)
