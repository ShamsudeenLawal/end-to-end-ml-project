from setuptools import setup, find_packages
from typing import List
from pathlib import Path

def get_requirements(filepath: str) -> List[str]:
    """Read the requirements from a file and return a list of packages."""
    with open(filepath, encoding="utf-8") as f:
        requirements =  [line.strip() for line in f if not line.startswith("#") and not line.startswith("-")]
    print(requirements)
    return requirements
    
setup(
    name="end-to-end-mlops",
    version="0.1.0",
    author="Shamsudeen Lawal",
    author_email="shamslaw564@gmail.com",
    description="An end-to-end MLOps project template",
    long_description="An end-to-end MLOps project template for building and deploying machine learning models.",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),)



# # Get the long description from the README file
# def readme():
#     with open("README.md", encoding="utf-8") as f:
#         return f.read()
# # Get the version from the __init__.py file
# def get_version():
#     version_file = Path("src") / "your_package" / "__init__.py"
#     with open(version_file, encoding="utf-8") as f:
#         for line in f:
#             if line.startswith("__version__"):
#                 return line.split("=")[1].strip().strip('"').strip("'")
#     raise RuntimeError("Unable to find version string.")
# # Get the requirements from the requirements.txt file
# def get_requirements():
#     with open("requirements.txt", encoding="utf-8") as f:
#         return [line.strip() for line in f if line.strip() and not line.startswith("#")]
# # Setup configuration
# setup(
#     name="your_package",
#     version=get_version(),
#     author="Your Name",
#     author_email="shamslaw564@gmail.com",
#     description="A brief description of your package",
#     long_description=readme(),
#     long_description_content_type="text/markdown",
#     url="",
#     packages=find_packages(where="src"),
#     package_dir={"": "src"},
#     install_requires=get_requirements(),
#     classifiers=[
#         "Programming Language :: Python :: 3",
#         "License :: OSI Approved :: MIT License",
#         "Operating System :: OS Independent",
#     ],
#     python_requires=">=3.6",
#     entry_points={
#         "console_scripts": [
#             "your_command=your_package.module:function",
#         ],
#     },
#     include_package_data=True,
#     zip_safe=False,
#     project_urls={
#         "Bug Tracker": "",
#         "Documentation": "",
#         "Source Code": "",
#         "Funding": "",
#         "Say Thanks!": "",
#         "Changes": "",
#         "Changelog": "",
#         "Release Notes": "",
#         "Contributors": "",
#         "Acknowledgements": "",
#         "License": "",
#         "Support": "",
#         "FAQ": "",
#         "Security": "",
#         "Translations": "",
#         "Testing": "",
#         "CI": "",
#         "CD": "",
#         "Deployment": "",
#         "Monitoring": "",
#         "Analytics": "",
#         "Performance": "",
#         "Scalability": "",
#         "Reliability": "",
#         "Availability": "",
#         "Maintainability": "",
#         "Portability": "",
#         "Interoperability": "",
#         "Usability": "",
#         "Accessibility": "",
#         "Internationalization": "",
#         "Localization": "",
#         "Documentation": "",
#         "Examples": "",
#         "Tutorials": "",
#         "Demos": ""})
