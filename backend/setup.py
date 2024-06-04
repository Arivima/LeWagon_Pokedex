from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='pokedex',
      version="0.0.10",
      description="Pokedex classification and generative models",
      license="MIT",
      author="TeamRocket",
      author_email="phdelville33@gmail.com",
      url="https://github.com/Arivima/LeWagon_Pokedex.git",
      install_requires=requirements,
      packages=find_packages(),
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)
