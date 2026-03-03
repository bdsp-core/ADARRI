from setuptools import setup, find_packages

setup(
    name="adarri",
    version="1.0.0",
    description="ADARRI: Detecting spurious R-peaks in ECG for HRV analysis",
    author="Dennis J. Rebergen, Sunil B. Nagaraj, Eric S. Rosenthal, "
           "Matt T. Bianchi, Michel J.A.M. van Putten, M. Brandon Westover",
    author_email="mwestover@mgh.harvard.edu",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "matplotlib>=3.7",
        "scikit-learn>=1.2",
        "neurokit2>=0.2.7",
        "pandas>=2.0",
        "h5py>=3.8",
        "tqdm>=4.65",
    ],
)
