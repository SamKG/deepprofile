# Deep Profile 

[![PyPI - Version](https://img.shields.io/pypi/v/deepprofile.svg)](https://pypi.org/project/deepprofile)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/deepprofile.svg)](https://pypi.org/project/deepprofile)

-----

This is a library that makes it easy to profile GPU usage for functions in Python. 

**Table of Contents**

- [Installation](#installation)
- [License](#license)

## Installation

1. Install deepprofile 
```console
pip install deepprofile 
```

## For nSight Systems
For nsight functionality, install Nvidia NSight Systems and make sure it's available in your PATH.

## For DCGM
For DCGM functionality, make sure you've installed DCGM. 

### Launching the DCGM daemon
This library includes a daemon which continuously outputs DCGM metrics to a CSV file. To see more information, run:
```console
python -m deepprofile --help
```


## License

`deepprofile` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
