from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='VarSVM',
    url='https://github.com/statmlben/VarSVM',
    author='Ben Dai',
    author_email='bdai@umn.edu',
    # Needed to actually package something
    packages=['VarSVM'],
    # Needed for dependencies
    install_requires=['numpy', 'scipy'],
    # *strongly* suggested for sharing
    version='0.1',
    # The license can be anything you like
    license='MIT',
    description='A python package for variant SVMs',
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)
