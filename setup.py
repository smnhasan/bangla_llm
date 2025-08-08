from setuptools import setup, find_packages

setup(
    name="bangla_llm",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0"
    ],
    author="S M Nahid Hasan",
    description="A Bangla LLM evaluation framework",
    url="https://github.com/smnhasan/bangla_llm",
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
    ],
)