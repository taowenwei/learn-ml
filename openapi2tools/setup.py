# setup.py

from setuptools import setup, find_packages

setup(
    name="openapi2tool",
    version="0.1",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'pyoat=openapi2tool.tool:main',
        ],
    },
    install_requires=[
        # Any dependencies your library requires
    ],
    author="Wenwei Tao",
    author_email="taowenwei@hotmail.com",
    description="A library to convert an OpenAPI spec to LangChain tools",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/openapi2tool",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
