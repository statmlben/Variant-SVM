from setuptools import setup, Extension
from Cython.Distutils import build_ext
import numpy as np

SRC_DIR = "VarSVM"
ext_1 = Extension(SRC_DIR + ".fastloop",
                  [SRC_DIR + "/fastloop.pyx"],
                  libraries=[],
                  include_dirs=[np.get_include()])


EXTENSIONS = [ext_1]

if __name__ == "__main__":
    setup(
        # Needed to silence warnings (and to be a worthwhile package)
        name='VarSVM',
        url='https://github.com/statmlben/Variant-SVM',
        author='Ben Dai',
        author_email='bdai@umn.edu',
        # Needed to actually package something
        packages=['VarSVM'],
        # Needed for dependencies
        install_requires=['numpy', 'scipy', 'Cython'],
        # *strongly* suggested for sharing
        version='0.2',
        download_url = 'https://github.com/statmlben/Variant-SVM/archive/0.2.tar.gz',
        # The license can be anything you like
        license='MIT',
        description='A python package for variant SVMs',
        cmdclass={"build_ext": build_ext},
        ext_modules=EXTENSIONS
        # We will also need a readme eventually (there will be a warning)
        # long_description=open('README.txt').read(),
    )
