#!/bin/bash

# Ask for video path once
read -p "Enter the path to the video: " video_path

while true; do
  echo
  echo "Choose execution method:"
  echo "1 - Sequential Method"
  echo "2 - OpenMP Method"
#  echo "3 - MPI Method"
  echo "0 - Exit"
  read -p "Your choice: " method

  if [ "$method" -eq 2 ]; then
    read -p "Enter number of OpenMP threads: " num_threads
    export OMP_NUM_THREADS="$num_threads"
  fi

#  if [ "$method" -eq 3 ]; then
#    read -p "Enter number of MPI processors: " num_procs
#  fi

  case $method in
    1)
      echo "Running Sequential_Method.exe..."
      export OPENCV_LOG_LEVEL=SILENT
      ./cmake-build-debug/Squential_Module.exe "$video_path"
      ;;
    2)
      echo "Running OpenMP_Method.exe with $num_threads threads..."
      export OPENCV_LOG_LEVEL=SILENT
      ./cmake-build-debug/OpenMP_Module.exe "$video_path"
      ;;
#    3)
#      echo "Running MPI_Module.exe with $num_procs processors..."
#      export OPENCV_LOG_LEVEL=SILENT
#      mpiexec -n "$num_procs" ./cmake-build-debug/MPI_Module.exe "$video_path"
#      ;;
    0)
      echo "Exiting..."
      break
      ;;
    *)
      echo "Invalid choice."
      ;;
  esac
done