from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'drone_agent'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='developer',
    maintainer_email='dev@example.com',
    description='AI-controlled drone with LLM brain and YOLO vision',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'brain_node = drone_agent.brain_node:main',
            'yolo_detector = drone_agent.yolo_detector:main',
        ],
    },
)
