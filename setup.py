from setuptools import find_packages,setup
from typing import List

hypen_e = '-e .'

def get_requirement(file_path:str)->List[str]:
  requirement =[]
  with  open(file_path) as file_obj:
    requirement = file_obj.readlines()
    requirement = [req.replace("\n"," ") for req in requirement]     
    if hypen_e in requirement:
      requirement.remove(hypen_e)
setup(

    name="CREDIT_CARD_FRAUD",
    version="0.0.1",
    author= "Abhishek",
    author_email="mritunjaygoswami5@gmail.com",
    packages=find_packages(),
    install_requires=  get_requirement('requirement.txt')
)