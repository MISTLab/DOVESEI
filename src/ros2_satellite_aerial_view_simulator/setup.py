from setuptools import setup
from glob import glob
import os
import platform

package_name = 'ros2_satellite_aerial_view_simulator'

python_version = ".".join(platform.python_version().split('.')[:2])

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, "aerialviewgenerator"],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/cfg', glob('cfg/*.json')),
        ('share/' + package_name, glob('launch/*.launch.py'))
    ],
    install_requires=['setuptools'],
    # setup_requires will download aerialviewgenerator from github during setup
    setup_requires=['AerialViewGenerator @ git+https://github.com/ricardodeazambuja/AerialViewGenerator.git'],
    # however, the directory won't be there while it's reading the info here, therefore it needs to be hardcoded
    package_dir = {
        "aerialviewgenerator": os.getcwd() + f"/.eggs/aerialviewgenerator-0.0.1-py{python_version}.egg/aerialviewgenerator/"
        },
    zip_safe=True,
    maintainer='ros2user',
    maintainer_email='ros2user@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'aerialimages_publisher = ros2_satellite_aerial_view_simulator.aerialimages:main',
        ],
    },
)
