import setuptools

with open("README.md","r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='libglcmsw',
    version='1.2.4',
    author="Simeon Atanasov",
    author_email="simeon.a.atanasov@gmail.com",
    despcription="A module that is able to produce sliding window image of large files",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/simo1427/libglcmsw",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Image Processing"
    ],
    python_requires='>3.7',
    keywords='glcm sliding window parallel computationalimaging',
    install_requires=['scikit-image','openslide-python']
)
