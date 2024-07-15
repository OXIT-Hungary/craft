import setuptools
from pathlib import Path

def get_install_requirements(path: Path):
    reqs = {}

    with open(path, "r", encoding="utf-8") as f:
        lines = [x.strip() for x in f.read().splitlines()]

    for req in [x for x in lines if not x.startswith("#")]:
        name = req
        for delim in ["==", ">=", "<="]:
            if delim in req:
                name, _ = req.split(delim)
                break

        reqs[name] = req

    return reqs

def get_long_description():
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()
    return long_description

setuptools.setup(
    name="craft",
    version="0.1",
    packages=setuptools.find_packages(where=Path(__file__).parent),
    python_requires=">=3.9",
    install_requires=list(get_install_requirements(Path(__file__).parent / "requirements.txt").values()),
    setup_requires=["wheel"],
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
)