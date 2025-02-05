from setuptools import setup, find_packages

_ = setup(
    name="pulsetrace",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
