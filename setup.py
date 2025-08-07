from setuptools import setup, find_packages


# Read requirements from requirements.txt if it exists
def read_requirements():
    try:
        with open('requirements.txt', 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        return []


# Core dependencies
install_requires = [
    # Core dependencies
    "pandas>=2.0.0",
    "numpy>=1.24.0",

    # LangChain ecosystem
    "langchain",
    "langchain_community",

    # ML/AI frameworks
    "InstructorEmbedding==1.0.1",
    "sentence-transformers==2.2.2",
    "transformers>=4.20",
    "torch>=2.0",

    # Data handling
    "datasets>=2.20",
    "pyarrow>=17.0",

    # Hugging Face
    "huggingface-hub==0.24.0",

    # Text processing
    "nltk>=3.8",

    # Additional utilities
    "tqdm>=4.64.0",
    "requests>=2.28.0",
]

# Optional dependencies
extras_require = {
    'gpu': [
        "llama-cpp-python[cublas]>=0.2.85",
    ],
    'dev': [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "black>=23.0.0",
        "flake8>=5.0.0",
        "mypy>=1.0.0",
        "pre-commit>=3.0.0",
    ],
    'docs': [
        "sphinx>=5.0.0",
        "sphinx-rtd-theme>=1.2.0",
        "myst-parser>=0.18.0",
    ]
}

# Add 'all' extra that includes everything
extras_require['all'] = sum(extras_require.values(), [])

setup(
    name="bangla_llm",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=install_requires,
    extras_require=extras_require,
    author="S M Nahid Hasan",
    author_email="smhasan.ruet.ece17@gmail.com",
    description="A Bangla LLM evaluation framework",
    long_description=open("README.md").read() if __name__ == "__main__" else "",
    long_description_content_type="text/markdown",
    url="https://github.com/smnhasan/bangla_llm",
    project_urls={
        "Bug Reports": "https://github.com/smnhasan/bangla_llm/issues",
        "Source": "https://github.com/smnhasan/bangla_llm",
        "Documentation": "https://bangla-llm.readthedocs.io/",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.9",
    keywords="bangla nlp llm evaluation machine-learning",
    include_package_data=True,
    zip_safe=False,
)