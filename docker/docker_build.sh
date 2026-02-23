#!/bin/bash

if [[ $# -lt 1 ]] ; then
  echo 'Arguments: tag_name'
  echo 'Example:   bash docker_build.sh martian-world-model'
  exit 1
fi

tagname=$1

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

docker build --rm \
  -f "${SCRIPT_DIR}/Dockerfile" \
  -t "${tagname}" \
  "${PROJECT_ROOT}"

echo "Successfully built Docker image with tag: ${tagname}"
