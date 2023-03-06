# `probemapper`

## Installation

The code was developed and tested on Linux.
The code may work on Mac and Windows system with minor modifications.

**1. Install ANTs**

To begin, you need to install the following softwares in your system.

* ANTs (For the installation, see https://github.com/ANTsX/ANTs/wiki/Compiling-ANTs-on-Linux-and-Mac-OS)

**2. Create venv and install**

```
python3 -m venv .env
source .env/bin/activate
pip install -U pip && pip install -r requirements.txt
pip install -e .
```

**2'. For advanced users: Python environment with mayavi**
To enable 3D visualization using mayavi, use the following commands

```
python3 -m venv .env
source .env/bin/activate
pip install -U pip && pip install -r requirements_mayavi.txt
pip install -e .
```
