from setuptools import setup

package_name = 'ros2_open_voc_landing_heatmap'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ros2user',
    maintainer_email='ros2user@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'getlandingheatmap_service = ros2_open_voc_landing_heatmap.generate_landing_heatmap:main',
            'lander_publisher = ros2_open_voc_landing_heatmap.lander_publisher:main',
        ],
    },
)
