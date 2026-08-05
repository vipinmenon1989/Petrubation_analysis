from setuptools import setup, find_packages

setup(
    name="pertps",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scanpy",
        "scipy",
        "statsmodels",
        "scikit-learn",
        "umap-learn",
        "matplotlib",
        "tqdm"
    ],
    author="Vipin Menon and Wei Li",
    description="Surgical Perturbation Score analysis for single-cell genomics",
    python_requires=">=3.8",
)
