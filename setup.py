import setuptools

with open("README.md","r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='libglcmsw',
    version='1.2.1dev',
    author="Simeon Atanasov",
    author_email="simeon.a.atanasov@gmail.com",
    despcription="A module that is able to produce sliding window image of large files",
    long_description=long_description,
    long_descroption_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: - :: -",
        "Operating System :: Linux",
    ],
    python_requires='>3.7',
    install_requires=['scikit-image','openslide-python']
)
