#!/bin/bash
# move into the parent directory
cd 'MSc(english) (WIP)'
echo "Building PDF of Master courses"
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
