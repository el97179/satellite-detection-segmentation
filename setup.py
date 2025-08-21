from setuptools import setup, find_packages

setup(
    name="satellite-detection-segmentation",
    version="0.1.0",
    description="Satellite Detection and Segmentation using CNN and Transformer models",
    author="tanman",
    author_email="el97179@yahoo.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "tensorflow>=2.13.0",
        "keras>=3.0.0",
        "transformers>=4.30.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.3.0",
        "mlflow>=2.4.0",
        "tqdm>=4.65.0",
        "pillow>=9.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.3.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
