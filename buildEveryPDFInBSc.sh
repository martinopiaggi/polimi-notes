#!/bin/bash

# move into the parent directory
cd 'BSc(italian)'

echo "Building PDF of Bachelor courses"

# iterate over each subdirectory
for directory in */; do
  cd "$directory"
  for script in *.sh; do
    if [ -f "$script" ]; then
      bash "$script"
    fi
  done
  cd ..
done
