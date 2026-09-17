import os
from glob import glob
from setuptools import find_packages, setup


package_name = "gs_nav_demo"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools", "PyYAML"],
    zip_safe=True,
    maintainer="zy",
    maintainer_email="2293086836@qq.com",
    description="Camera augmented-reality navigation overlay demo for ROS 2.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "ar_nav_node = gs_nav_demo.ar_nav_node:main",
            "qt_nav_node = gs_nav_demo.qt_nav_node:main",
        ],
    },
)
