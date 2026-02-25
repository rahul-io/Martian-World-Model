#!/bin/bash

if [[ $# -lt 1 ]] ; then
  echo 'Arguments: tag_name'
  echo 'Example:   bash docker_run.sh martian-world-model'
  exit 1
fi

tagname=$1

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

xhost +local:docker

docker run \
  --rm -ti \
  --gpus all \
  -e DISPLAY=$DISPLAY \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e GITHUB_TOKEN=$GITHUB_TOKEN \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "${PROJECT_ROOT}":/workspace/Martian-World-Model \
  -e QT_X11_NO_MITSHM=1 \
  --privileged \
  --shm-size=16g \
  --network=host \
  --name martian_world_model_container \
  --workdir /workspace/Martian-World-Model \
  "${tagname}"
