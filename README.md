# CLS-Scientific_Computing-Assignment2
The DLA model simulates growth via diffusion, selecting growth sites based on concentration. A Monte Carlo version uses random walkers that stick upon contact. The Gray-Scott model simulates reaction-diffusion of chemicals U and V, forming patterns based on reaction rates. Different parameters influence growth shapes and dynamics in both models.

![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Table of Contents

- [Description](#description)
- [Getting Started](#getting-started)
- [File Descriptions](#file-descriptions)
- [Contributors](#contributors)
- [Git Fame](#git-fame)
- [License](#license)

## Description

Within this repository simulations are created for the two versions of DLA and Gray-Scott models.

## Getting Started

### Installation
First clone the repository.
```bash
git clone https://github.com/kingilsildor/CLS-Scientific_Computing-Assignment2
cd repository
```

### Prerequisites

To get the project running, install all the packages from the installer.
For this the following command can be used:
```bash
# Example
pip install -r requirements.txt
```

### Interface
Different modules can be run separately from their file.
But the main inferface for the project is `interface.ipynb` in the root folder.
This file uses all the functions that are important to run the code.

### Style Guide
For controbuting to this project it is important to know the style used in this document.
See the [STYLEGUIDE](STYLEGUIDE.md) file for details.


## File Descriptions

| File/Folder | Description |
|------------|-------------|
| `interface.ipynb` | Interface for all the code |
| `modules/config.py` | File for all constants |
| `modules/dla_algorithm.py` | File for simulation of DLA using growth probabilities based on concentration |
| `modules/gray_scott.py` | File for simulation of Gray-Scott model |
| `modules/grid.py` | File containing basic grid functions |
| `modules/random_walk_monte_carlo.py` | File for simulation of DLA using Monte Carlo random walk |
| `data/*` | Store for the data that the functions will write |
| `results/*`| Images and animations of the files |

## Contributors

List all contributors to the project.

- [Tycho Stam](https://github.com/kingilsildor)
- [Anezka Potesilova](https://github.com/anezkap)
- [Michael MacFarlane Glasow](https://github.com/mdmg01)

## Git Fame

Total commits: XX
Total ctimes: XX
Total files: XX
Total loc: XX
| Author            |   loc |   coms |   fils |  distribution   |
|:------------------|------:|-------:|-------:|:----------------|


Note: Tycho Stam -> kingilsildor

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.