from setuptools import setup, find_packages

setup(
    name="marl-traffic-control",
    version="1.0.0",
    author="Your Name",
    description="Multi-Agent RL for Smart Traffic Signal Control",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "gymnasium>=0.29.0",
        "pyyaml>=6.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tensorboard>=2.13.0",
        "tqdm>=4.65.0",
        "scikit-learn>=1.3.0",
    ],
)