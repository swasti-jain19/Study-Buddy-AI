from setuptools import setup,find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="Study Buddy AI",
    version="0.1",
    author="Swasti Jain",
    packages=find_packages(),
    install_requires = requirements,
)