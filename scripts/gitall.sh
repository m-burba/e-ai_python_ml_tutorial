#!/bin/bash

USAGE=$(cat << EOF
################################################################################
#
# Script gitall to check all git folders located in the current folder
#
# USAGE: ./gitall.sh [-h] <pull|-> [s] 
#     ./gitall.sh pull 
#         updates all repositories in folder
#     ./gitall.sh s
#         shows the git status of all repositories in folder
#     ./gitall.sh pull s
#         pulls and then shows the status of all repositories in folder
#
# Options:
#     -h          Show this help message and exit
#
# Recommendation: 
#   Set a soft link like: 
#         ln -s ~/dace_play/scripts/gitall.sh gitall
#   into your ~/bin or \$WORK/bin folder, then you can call gitall everywhere
#   with the above options. 
# 
# Author: 
#    2020 Roland Potthast initial version 
#   
################################################################################
EOF
)

################################################################################
# some preparation
################################################################################
pad=".............................................................................."
pad2="                                                                             "
pad3="\r| \r\t\t\t   | \r\t\t\t\t\t\t |"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
DARKBLUE='\033[0;34m'
CYAN='\033[0;36m'
GREY='\033[0;37m'
VIOLET='\033[38;5;93m'
NC='\033[0m' # No Color
DARKGREY='\033[90m'

################################################################################
# checking options
################################################################################
opt_d=0
while getopts :hd options; do
  case "$options" in
    h) echo -e "$USAGE"; exit 0;;
    *) echo -e "${RED}Invalid option: -$OPTARG${NC}\n"; echo -e "$USAGE"; exit 1;;
  esac
done
shift $((OPTIND-1))

################################################################################
# Now checking git folders
################################################################################

echo -e "${DARKGREY}$pad${NC}" 
echo -e "${DARKGREY}Updating and checking status of all git repositories in folder${NC}"
echo -e "${DARKGREY}$pad \rGit Repository \r\t\t\t * Branch \r\t\t\t\t\t\t Origin or Status${NC}" 
echo -e "${DARKGREY}$pad2 $pad3${NC}"
in1="$1" # for later, will be modified
in2="$2" # for later, will be modified
jc=0 
for j in */; do
  mywd=$(pwd)
  if cd "$j" 2> /dev/null; then
    if [ -d .git ]; then 
      repo_name=$(basename "$j")
      branch_info=$(git branch --show-current)
      upstream_info=$(git rev-parse --abbrev-ref --symbolic-full-name "@{u}" 2>/dev/null)

      if [ -z "$upstream_info" ]; then
        echo -e "${GREY}$pad \r$repo_name \r\t\t\t $branch_info \r\t\t\t\t\t\t No upstream branch set${NC}"
      else
        if [ "$in1" == "pull" ]; then 
          pull_result=$(git pull 2>&1)
          if [[ "$pull_result" == *"Already up to date."* ]]; then
            pull_status="${GREEN}Already up to date.${NC}"
          elif [[ "$pull_result" == *"error"* ]]; then
            pull_status="${RED}Error during pull.${NC}"
          else
            pull_status="${VIOLET}Updated.${NC}"
          fi
        else
          pull_status=$(git remote get-url origin)
        fi

        echo -e "${DARKBLUE}$pad \r$repo_name \r\t\t\t * $branch_info \r\t\t\t\t\t\t $pull_status${NC}"

        if [[ "$in2" == "s" || "$in1" == "s" ]]; then
          git status -s
        fi
      fi

      jc=$(( jc + 1 )) 
      if [ $(( jc % 5 )) == 0 ]; then
        echo -e "${DARKGREY}$pad2 $pad3${NC}"
      fi
    fi
    cd "$mywd"
  fi

done
