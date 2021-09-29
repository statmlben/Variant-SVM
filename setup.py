from setuptools import setup, Extension
# from Cython.Distutils import build_ext
import numpy as np

SRC_DIR = "varsvm"
ext_1 = Extension(SRC_DIR + ".fastloop",
                  [SRC_DIR + "/fastloop.pyx"],
                  libraries=[],
                  include_dirs=[np.get_include()])

# with open('README.rst') as f:
#     LONG_DESCRIPTION = f.read()

EXTENSIONS = [ext_1]

if __name__ == "__main__":
    setup(
        # Needed to silence warnings (and to be a worthwhile package)
        name='varsvm',
        description='Python library for Variants of Support Vector Machines',
        url='https://github.com/statmlben/Variant-SVM',
        author='Ben Dai',
        author_email='bendai@cuhk.edu.hk',
        # Needed to actually package something
        packages=['varsvm'],
        # Needed for dependencies
        install_requires=['numpy', 'scipy', 'Cython'],
        # *strongly* suggested for sharing
        version='1.4',
        download_url = 'https://github.com/statmlben/Variant-SVM/archive/1.4.tar.gz',
        # The license can be anything you like
        license='MIT',
        #cmdclass={"build_ext": build_ext},
        ext_modules=EXTENSIONS,
        # We will also need a readme eventually (there will be a warning)
        # long_description=LONG_DESCRIPTION
    )