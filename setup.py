from setuptools import find_packages, setup

setup(
    name='MithunPathRule',
    version='1',
    description='TBC',
    url='None',
    packages=['pathrule'],
    author='Mithun Vaidhyanathan',
    install_requires=[
        'bandit',
    ],
    entry_points={
        'bandit.plugins': [
            #'mypathrulestr = pathrule.using_path_type:is_path_there_str',
            'mypathrulecall = pathrule.using_path_type:is_path_there_call',
        ],
    }
)