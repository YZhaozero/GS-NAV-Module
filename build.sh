#!/usr/bin/env bash

set -euo pipefail

workspace_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace_install="${workspace_root}/install"
dependencies_root="${GS_NAV_DEPS_ROOT:-${HOME}/gs_nav_dependencies}"
dependencies_setup="${dependencies_root}/setup.bash"

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
if [[ -r "${dependencies_setup}" ]]; then
  source "${dependencies_setup}"
else
  printf '%s\n' \
    "[GS-NAV] Warning: external dependencies setup was not found:" \
    "         ${dependencies_setup}" \
    "         Run ${workspace_root}/scripts/install_dependencies.sh first."
fi
set -u
cd "${workspace_root}"

build_arguments=(
  --symlink-install
  --allow-overriding
    realsense2_camera
    realsense2_camera_msgs
    realsense2_description
)

# These packages are copied and built as an external underlay by the dependency
# installer. Ignore the vendored copies here so their artifacts do not end up in
# this workspace. Set GS_NAV_BUILD_BUNDLED_THIRDPARTY=1 only for offline fallback.
if [[ "${GS_NAV_BUILD_BUNDLED_THIRDPARTY:-0}" != "1" ]]; then
  build_arguments+=(
    --packages-ignore
      fast_gicp
      ndt_omp_ros2
      scancontext_ros2
  )
fi

printf '%s\n' "[GS-NAV] Building workspace: ${workspace_root}"
printf '%s\n' "[GS-NAV] Dependency root:   ${dependencies_root}"

exec colcon build "${build_arguments[@]}" "$@"
