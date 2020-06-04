#!/usr/bin/env python3


import setuptools
import os


def _read_reqs(relpath):
    fullpath = os.path.join(os.path.dirname(__file__), relpath)
    with open(fullpath) as f:
        return [
            s.strip() for s in f.readlines() if (s.strip() and not s.startswith("#"))
        ]


_REQUIREMENTS_TXT = _read_reqs("requirements.txt")
_INSTALL_REQUIRES = [l for l in _REQUIREMENTS_TXT if "://" not in l]

setuptools.setup(
    name="ada",
    version="0.1",
    install_requires=["Cython", "numpy>=1.16.4"]
    + _INSTALL_REQUIRES
    + [
        #                         'imputation @ git+git@gitlab.criteois.com:am.tousch/imputation.git@29a1cac8f55e3b1fb881991e590acaac2a2e5abe',
        #                         'POT @ git+https://github.com/rflamary/POT.git@abfe183a49caaf74a07e595ac40920dae05a3c22',
        #                        'torch_salad @ git+https://github.com/domainadaptation/salad.git@5f55d6fb2ab0f55191050382db91f19ebac675e0',
    ],
    data_files=[(".", ["requirements.txt"])],
    packages=setuptools.find_packages(),
    package_dir={"": "."},
)
