from glob import glob

from setuptools import find_packages, setup
from setuptools import Extension
import subprocess

package_name = 'orbbec_bev_odometry'


def pcl_extension():
    cflags = subprocess.check_output(
        ['pkg-config', '--cflags', 'pcl_registration'], text=True).split()
    libs = subprocess.check_output(
        ['pkg-config', '--libs', 'pcl_registration'], text=True).split()
    include_dirs = [flag[2:] for flag in cflags if flag.startswith('-I')]
    library_dirs = [flag[2:] for flag in libs if flag.startswith('-L')]
    libraries = [flag[2:] for flag in libs if flag.startswith('-l')]
    return Extension(
        package_name + '._pcl_gicp',
        sources=['orbbec_bev_odometry/pcl_gicp.cpp'],
        include_dirs=include_dirs + ['/usr/include/pybind11'],
        library_dirs=library_dirs, libraries=libraries,
        language='c++', extra_compile_args=['-O2', '-std=c++17'],
    )

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    ext_modules=[pcl_extension()],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='oit',
    maintainer_email='oit@todo.todo',
    description='BEV odometry node for an Orbbec Femto Bolt depth camera.',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'bev_odometry_node = orbbec_bev_odometry.bev_odometry_node:main',
            'pointcloud_odometry_node = orbbec_bev_odometry.pointcloud_odometry_node:main',
            'pointcloud_offline_ab = orbbec_bev_odometry.offline_ab:main',
        ],
    },
)
