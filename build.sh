#!/usr/bin/env bash

set -euo pipefail

workspace_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace_install="${workspace_root}/install"

# A terminal that has already sourced this workspace treats the previous
# install as an underlay during the next build. Remove only this workspace's
# entries while preserving ROS and any external underlays.
strip_workspace_install() {
  local variable_name="$1"
  local current_value="${!variable_name-}"
  local cleaned_value=""
  local entry

  IFS=':' read -r -a entries <<< "${current_value}"
  for entry in "${entries[@]}"; do
    case "${entry}" in
      "${workspace_install}"|"${workspace_install}/"*)
        continue
        ;;
    esac
    if [[ -n "${entry}" ]]; then
      cleaned_value="${cleaned_value:+${cleaned_value}:}${entry}"
    fi
  done

  if [[ -n "${cleaned_value}" ]]; then
    export "${variable_name}=${cleaned_value}"
  else
    unset "${variable_name}"
  fi
}

for path_variable in \
  AMENT_PREFIX_PATH \
  CMAKE_PREFIX_PATH \
  COLCON_PREFIX_PATH \
  LD_LIBRARY_PATH \
  PKG_CONFIG_PATH \
  PYTHONPATH \
  PATH; do
  strip_workspace_install "${path_variable}"
done

# ROS Humble's generated setup scripts may inspect unset optional variables.
set +u
source /opt/ros/humble/setup.bash
set -u
cd "${workspace_root}"

exec colcon build \
  --symlink-install \
  --allow-overriding \
    realsense2_camera \
    realsense2_camera_msgs \
    realsense2_description \
  "$@"
