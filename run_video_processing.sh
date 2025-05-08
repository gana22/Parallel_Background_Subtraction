#!/bin/bash

# Define color codes and styles
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Clear screen and print a nice ASCII-styled title
clear
echo -e "${CYAN}${BOLD}"
echo "                 ╔════════════════════════════════════════════════╗"
echo "                 ║           BACKGROUND SUBTRACTION TOOL          ║"
echo "                 ╚════════════════════════════════════════════════╝"
echo -e "${NC}"

# Ask for video path
echo -e "${YELLOW}📁 Please enter the path to the video file:${NC}"
read -p "➡️  " video_path

# Main menu loop
while true; do
  echo
  echo -e "${BLUE}=============================================${NC}"
  echo -e "${BOLD}${CYAN}🔧 Choose Execution Method:${NC}"
  echo -e "      ${GREEN}[1]${NC} Sequential Method"
  echo -e "      ${GREEN}[2]${NC} OpenMP Method"
# echo -e "      ${GREEN}[3]${NC} MPI Method"
  echo -e "      ${GREEN}[0]${NC} Exit"
  echo -e "${BLUE}=============================================${NC}"
  read -p "$(echo -e "${YELLOW}Your choice:${NC} ")" method
  echo

  # Handle OpenMP input
  if [ "$method" -eq 2 ]; then
    read -p "$(echo -e "${YELLOW}🔢 Enter number of OpenMP threads:${NC} ")" num_threads
    export OMP_NUM_THREADS="$num_threads"
  fi

# Handle MPI input (optional)
# if [ "$method" -eq 3 ]; then
#   read -p "$(echo -e "${YELLOW}🔢 Enter number of MPI processors:${NC} ")" num_procs
# fi

  # Execute based on user's choice
  case $method in
    1)
      echo -e "${GREEN}🚀 Running Sequential_Module.exe...${NC}"
      export OPENCV_LOG_LEVEL=SILENT
      ./cmake-build-debug/Squential_Module.exe "$video_path"
      ;;

    2)
      echo -e "${GREEN}🚀 Running OpenMP_Module.exe with ${num_threads} threads...${NC}"
      export OPENCV_LOG_LEVEL=SILENT
      ./cmake-build-debug/OpenMP_Module.exe "$video_path"
      ;;

#   3)
#     echo -e "${GREEN}🚀 Running MPI_Module.exe with ${num_procs} processors...${NC}"
#     export OPENCV_LOG_LEVEL=SILENT
#     mpiexec -n "$num_procs" ./cmake-build-debug/MPI_Module.exe "$video_path"
#     ;;

    0)
      # Show exit message and wait for user input before breaking
      echo -e "${CYAN}👋 Exiting... Have a great day!"
      read -p "$(echo -e "${YELLOW}Press Enter to exit...")"
      break
      ;;

    *)
      echo -e "${RED}❌ Invalid choice. Please enter 0, 1, or 2."
      ;;
  esac
done
