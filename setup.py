from pathlib import Path
from setuptools import setup, find_packages

here = Path(__file__).resolve().parent

requirements_path = here / "requirements.txt"
if requirements_path.exists():
    requirements = [
        line.strip()
        for line in requirements_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
else:
    requirements = []

readme = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="GPT_tools",              # distribution name (pip install ...)
    version="0.1.0",               # no leading 'v'
    packages=find_packages(),      # remove package_dir unless using src layout
    url="https://github.com/AdamCBartnik/GPT_tools",
    long_description=readme,
    long_description_content_type="text/markdown",
    install_requires=requirements,
    include_package_data=True,
    python_requires=">=3.8",       # or whatever you actually support
)

