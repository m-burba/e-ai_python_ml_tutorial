#!/bin/bash

# Loop through chapters 01 to 18
for i in $(seq -w 1 18)
do
  # Filename
  file="ch${i}.tex"

  # Write chapter heading to file
  echo "\\chapter{Chapter${i}}" > "$file"

  # Loop through 4 sections per chapter
  for j in $(seq -w 1 4)
  do
    echo "\\section{Section${i}-${j}}" >> "$file"
  done
done

