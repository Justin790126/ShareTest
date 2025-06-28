from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import os

# Set the correct source and package directories for proper module visibility
src_path = os.path.join(os.path.dirname(__file__), 'src')

extensions = [
    Extension(
        name="libtest",
        sources=[os.path.join("src", "libtest.pyx")],
        include_dirs=[numpy.get_include()],
        language="c",
    )
]

setup(
    name="libtest",
    ext_modules=cythonize(extensions),
    install_requires=["numpy"],
    zip_safe=False,
    package_dir={
        'layer1': os.path.join('src', 'layer1'),
        'layer2': os.path.join('src', 'layer2'),
        'layer3': os.path.join('src', 'layer3'),
    },
    packages=['layer1', 'layer2', 'layer3'],
)