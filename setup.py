from glob import glob

from setuptools import setup

package_name = "hier_robot"

setup(
    name=package_name,
    version="1.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*")),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/worlds", glob("worlds/*.world")),
        ("share/" + package_name + "/maps", glob("maps/*")),
    ],
    install_requires=["setuptools", "pyyaml", "numpy", "scipy"],
    zip_safe=True,
    maintainer="Mati",
    description="Hierarchical robot: LLM planner -> D* Lite -> LQR, on RTAB-Map SLAM with a SigLIP2 semantic tensor",
    license="MIT",
    entry_points={
        "console_scripts": [
            "robot_node = hier_robot.robot_node:main",
            "mapper_node = hier_robot.mapper_node:main",
        ],
    },
)
