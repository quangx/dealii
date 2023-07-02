#!/bin/bash
set -eu

DEAL_II_SOURCE_DIR="${1}"
shift
CMAKE_CURRENT_SOURCE_DIR="${1}"
shift

#
# use .diff files to create .cc files
#

# Find all *.diff files in the current directory
diff_files="$(find ${CMAKE_CURRENT_SOURCE_DIR} -maxdepth 1 -type f -name "*.diff")"

for file in ${diff_files}; do
  # Extract the filename without the extension
  filename="$(basename "${file}" ".diff")"
  source_file="${DEAL_II_SOURCE_DIR}/examples/${filename}/${filename}.cc"
  output_files="${CMAKE_CURRENT_SOURCE_DIR}/${filename}*.output*"

  # Check if the corresponding .cc file exists
  if [[ -f "${source_file}" ]]; then
    # Apply the diff and save it to a new file
    patch "${source_file}" -o "${filename}.cc" < "${file}"
    echo "copying file(s) ${output_files}"
    cp ${output_files} .
  else
    echo "No matching .cc file found for ${file}"
    exit 1
  fi
done
