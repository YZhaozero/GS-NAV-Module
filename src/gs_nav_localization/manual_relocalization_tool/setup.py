from setuptools import find_packages, setup

package_name = 'manual_relocalization_tool'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zy',
    maintainer_email='2293086836@qq.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'manual_reloc_node = manual_relocalization_tool.manual_reloc_node:main',
        ],
    },
)
