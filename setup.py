from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="nightmarenet",
    version="0.2.0",
    author="Adit Jain",
    description="A Sleep-Inspired Training Paradigm for AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Adit-Jain-srm/NightmareNet",
    packages=find_packages(include=["nightmarenet*", "scripts*"]),
    python_requires=">=3.9",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "nightmarenet-train=scripts.train:main",
            "nightmarenet-generate=scripts.generate_data:main",
            "nightmarenet-evaluate=scripts.evaluate:main",
        ],
    },
)
