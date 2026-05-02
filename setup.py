from setuptools import setup, find_packages

setup(
    name="driftmonitor",
    version="2.0.0",
    description="Automated modular data drift monitoring for ML models",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "scipy>=1.9.0",
        "matplotlib>=3.6.0",
        "streamlit>=1.28.0",
        "streamlit-autorefresh>=0.0.1",
    ],
    python_requires=">=3.8",
)
