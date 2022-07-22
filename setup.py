from setuptools import setup, find_packages

setup(
    name='iss_preprocess',
    version='v0.1',
    url='https://github.com/znamlab/iss-preprocess',
    license='MIT',
    author='Znamenskiy lab',
    author_email='petr.znamenskiy@crick.ac.uk',
    description='Tools for processing in situ sequencing data',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'matplotlib',
        'scikit-image',
        'czifile',
        'scikit-learn',
        'pystackreg',
        'flexiznam @ git+ssh://git@github.com/znamlab/flexiznam.git',
        'trackpy',
        'numba',
        'cellpose'
    ],
    entry_points='''
        [console_scripts]
        ''',
)
