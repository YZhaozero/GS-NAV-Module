#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace_root="$(cd -- "${script_dir}/.." && pwd)"
dependencies_root="${GS_NAV_DEPS_ROOT:-${HOME}/gs_nav_dependencies}"
jobs="${GS_NAV_BUILD_JOBS:-$(nproc)}"
install_apt=1

usage() {
  cat <<'EOF'
Usage: ./scripts/install_dependencies.sh [options]

Install GS-NAV system and source dependencies for Ubuntu 22.04 / ROS 2 Humble.
Source packages, build trees, and user-installed artifacts are kept outside the
GS-NAV workspace (default: ~/gs_nav_dependencies). GTSAM 4.2 is installed as
system packages from the official BorgLab Launchpad PPA.

Options:
  --deps-root PATH  External dependency directory
  --jobs N          Parallel build jobs (default: number of CPU cores)
  --skip-apt        Do not run apt/rosdep (useful after the first install)
  -h, --help        Show this help

Environment equivalents:
  GS_NAV_DEPS_ROOT, GS_NAV_BUILD_JOBS
EOF
}

while (($#)); do
  case "$1" in
    --deps-root)
      [[ $# -ge 2 ]] || { echo "--deps-root requires a path" >&2; exit 2; }
      dependencies_root="$2"
      shift 2
      ;;
    --jobs)
      [[ $# -ge 2 ]] || { echo "--jobs requires a number" >&2; exit 2; }
      jobs="$2"
      shift 2
      ;;
    --skip-apt)
      install_apt=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ "${jobs}" =~ ^[1-9][0-9]*$ ]] || {
  echo "Build jobs must be a positive integer: ${jobs}" >&2
  exit 2
}

dependencies_root="$(realpath -m -- "${dependencies_root}")"
case "${dependencies_root}" in
  /|"${HOME}"|"${workspace_root}"|"${workspace_root}"/*)
    echo "Refusing unsafe dependency directory: ${dependencies_root}" >&2
    echo "Choose a dedicated directory outside the GS-NAV workspace." >&2
    exit 2
    ;;
esac

source_root="${dependencies_root}/src"
build_root="${dependencies_root}/build"
native_install="${dependencies_root}/install"
ros_workspace="${dependencies_root}/ros2_ws"
marker="${dependencies_root}/.gs_nav_dependencies"

if [[ -e "${dependencies_root}" && ! -f "${marker}" ]]; then
  echo "The target exists but is not managed by this installer:" >&2
  echo "  ${dependencies_root}" >&2
  echo "Use an empty path or set GS_NAV_DEPS_ROOT to another directory." >&2
  exit 2
fi

mkdir -p "${source_root}" "${build_root}" "${native_install}" \
  "${ros_workspace}/src"
touch "${marker}"

# A user may invoke this script from a terminal that already sourced either
# this workspace or an older dependency underlay. Remove only those prefixes so
# colcon does not treat stale copies of the same packages as an underlay.
strip_managed_prefixes() {
  local variable_name="$1"
  local current_value="${!variable_name-}"
  local cleaned_value=""
  local entry

  IFS=':' read -r -a entries <<< "${current_value}"
  for entry in "${entries[@]}"; do
    case "${entry}" in
      "${workspace_root}/install"|"${workspace_root}/install/"*|\
      "${ros_workspace}/install"|"${ros_workspace}/install/"*|\
      "${native_install}"|"${native_install}/"*)
        continue
        ;;
    esac
    [[ -z "${entry}" ]] || \
      cleaned_value="${cleaned_value:+${cleaned_value}:}${entry}"
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
  strip_managed_prefixes "${path_variable}"
done

if ((install_apt)); then
  echo "[1/6] Configuring the BorgLab GTSAM 4.2 PPA..."
  sudo apt-get update
  sudo apt-get install -y software-properties-common
  sudo add-apt-repository -y ppa:borglab/gtsam-release-4.2
  sudo apt-get update

  echo "[1/6] Installing GTSAM, Ubuntu, and ROS dependencies..."
  sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    libboost-all-dev \
    libeigen3-dev \
    libfmt-dev \
    libgtsam-dev \
    libgtsam-unstable-dev \
    libgoogle-glog-dev \
    libopencv-dev \
    libpcl-dev \
    libssl-dev \
    libusb-1.0-0-dev \
    libyaml-cpp-dev \
    iputils-ping \
    ninja-build \
    pkg-config \
    python3-colcon-common-extensions \
    python3-rosdep \
    ripgrep \
    rsync \
    usbutils \
    vim

  if [[ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]]; then
    sudo rosdep init
  fi
  rosdep update

  # Sophus is built below. GTSAM comes from the BorgLab 4.2 PPA above, rather
  # than ros-humble-gtsam. cmake_modules is a stale ROS 1 manifest dependency;
  # the ROS 2 build keeps USE_ROS=False and uses its own FindEigen.cmake.
  source /opt/ros/humble/setup.bash
  rosdep install \
    --from-paths "${workspace_root}/src" \
    --ignore-src \
    --rosdistro humble \
    --skip-keys "sophus gtsam cmake_modules" \
    -r -y
else
  echo "[1/6] Skipping apt and rosdep as requested."
fi

clone_at() {
  local name="$1"
  local url="$2"
  local revision="$3"
  local target="${source_root}/${name}"

  if [[ ! -e "${target}" ]]; then
    git clone --recursive "${url}" "${target}"
  elif [[ ! -d "${target}/.git" ]]; then
    echo "Dependency source exists but is not a Git checkout: ${target}" >&2
    exit 2
  fi

  git -C "${target}" fetch --tags origin
  git -C "${target}" checkout --detach "${revision}"
  git -C "${target}" submodule sync --recursive
  git -C "${target}" submodule update --init --recursive
}

cmake_install() {
  local name="$1"
  shift
  cmake \
    -S "${source_root}/${name}" \
    -B "${build_root}/${name}" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${native_install}" \
    "$@"
  cmake --build "${build_root}/${name}" --parallel "${jobs}"
  cmake --install "${build_root}/${name}"
}

echo "[2/6] Installing Sophus 1.22.10..."
clone_at Sophus https://github.com/strasdat/Sophus.git 1.22.10
cmake_install Sophus \
  -DSOPHUS_INSTALL=ON \
  -DBUILD_SOPHUS_TESTS=OFF \
  -DBUILD_SOPHUS_EXAMPLES=OFF

echo "[3/6] Installing Livox-SDK2 v1.2.5..."
clone_at Livox-SDK2 https://github.com/Livox-SDK/Livox-SDK2.git v1.2.5
cmake_install Livox-SDK2

echo "[4/6] Installing the tested TEASER++ revision..."
clone_at TEASER-plusplus \
  https://github.com/MIT-SPARK/TEASER-plusplus.git \
  e2f6f17
cmake_install TEASER-plusplus \
  -DBUILD_TESTING=OFF \
  -DBUILD_DOC=OFF \
  -DBUILD_PYTHON_BINDINGS=OFF

prepend_path() {
  local variable_name="$1"
  local value="$2"
  local current_value="${!variable_name-}"
  export "${variable_name}=${value}${current_value:+:${current_value}}"
}

echo "[5/6] Building ROS 2 third-party packages in the external underlay..."
for package in fast_gicp ndt_omp_ros2 scancontext_ros2; do
  source_package="${workspace_root}/src/gs_nav_localization/thirdrepo/${package}"
  target_package="${ros_workspace}/src/${package}"
  [[ -d "${source_package}" ]] || {
    echo "Missing vendored dependency source: ${source_package}" >&2
    exit 2
  }
  mkdir -p "${target_package}"
  rsync -a --delete "${source_package}/" "${target_package}/"
done

set +u
source /opt/ros/humble/setup.bash
set -u
prepend_path CMAKE_PREFIX_PATH "${native_install}"
prepend_path LD_LIBRARY_PATH "${native_install}/lib"
prepend_path PKG_CONFIG_PATH "${native_install}/lib/pkgconfig"

(
  cd "${ros_workspace}"
  colcon build \
    --merge-install \
    --packages-select fast_gicp ndt_omp_ros2 scancontext_ros2 \
    --parallel-workers "${jobs}" \
    --cmake-args -DCMAKE_BUILD_TYPE=Release
)

echo "[6/6] Writing reusable environment setup..."
setup_file="${dependencies_root}/setup.bash"
{
  printf '%s\n' '#!/usr/bin/env bash'
  printf 'export GS_NAV_DEPS_ROOT=%q\n' "${dependencies_root}"
  printf '%s\n' 'source /opt/ros/humble/setup.bash'
  printf 'export CMAKE_PREFIX_PATH=%q:"${CMAKE_PREFIX_PATH:-}"\n' "${native_install}"
  printf 'export LD_LIBRARY_PATH=%q/lib:"${LD_LIBRARY_PATH:-}"\n' "${native_install}"
  printf 'export PKG_CONFIG_PATH=%q/lib/pkgconfig:"${PKG_CONFIG_PATH:-}"\n' "${native_install}"
  printf 'source %q\n' "${ros_workspace}/install/setup.bash"
} > "${setup_file}"
chmod +x "${setup_file}"

test -r "${native_install}/include/livox_lidar_api.h"
test -r "${native_install}/share/sophus/cmake/SophusConfig.cmake"
test -r "${native_install}/lib/cmake/teaserpp/teaserppConfig.cmake"
test -r "${ros_workspace}/install/setup.bash"

echo
echo "GS-NAV dependencies are ready in: ${dependencies_root}"
echo "Next steps:"
echo "  source ${setup_file}"
echo "  ${workspace_root}/build.sh"
