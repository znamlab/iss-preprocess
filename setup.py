from setuptools import find_packages, setup

setup(
    name="iss_preprocess",
    version="v0.2.3",
    url="https://github.com/znamlab/iss-preprocess",
    license="MIT",
    author="Znamenskiy lab",
    author_email="petr.znamenskiy@crick.ac.uk",
    description="Tools for processing in situ sequencing data",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "matplotlib",
        "scikit-image",
        "scikit-learn",
        "flexiznam @ git+ssh://git@github.com/znamlab/flexiznam.git",
        "znamutils @ git+ssh://git@github.com/znamlab/znamutils.git",
        "image-tools @ git+ssh://git@github.com/znamlab/image-tools.git",
        "opencv-python",
        "numba",
        "cellpose",
        "Click",
        "brainglobe_atlasapi",
        "natsort",
        "seaborn",
        "decorator",
        "scipy>=1.11.0",
    ],
    entry_points="""
        [console_scripts]
        iss=iss_preprocess.cli.iss:iss_cli
        iss-register=iss_preprocess.cli.register_cli:register_cli
        iss-sync-and-crunch=iss_preprocess.cli.sync_and_crunch_cli:sync_and_crunch_cli
        """,
)
