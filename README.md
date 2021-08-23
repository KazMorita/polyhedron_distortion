![image](image_distortion.png)

# Overview
Polyhedron distortion analysis using group theory.
This could either be used as a standalone script or as API.

# Background
This code allows you to convert distortions into a small-sized vector, possibly suitable for machine learning input.
Example for octahedron is given, but it is applicable to any type of polyhedra.

# Contents
## Dependencies
- numpy
- phonopy (developed with version 2.8)
- pymatgen (developed with version 2022.0.8)

## Installation
The two scripts `basis_generator.py` and `polyhedron_analysis.py` does not rely on each other.
Installation procedure is same as other python scripts.

If you want to use it as a script, you can simply execute it.
If you want to use the API, you could either:
- export PYTHONPATH=\<full path to the polyhedron_distortion directory\>:$PYTHONPATH
- place the files in the same directory as your script

# Usage
## As a script
- ### creating basis sets
```
python basis_generator.py
```
This will create json file inside the `basis` directory.
The repository already includes this output.
- ### obtaining four dimensional vector
```
python polyhedron_analysis.py POSCAR n
```
where POSCAR is [VASP](https://www.vasp.at/) POSCAR and n is n-th atom in the centre of the octahedron.
Use the API for other input types.

## As an API
- see [Tutorial notebook](https://github.com/KazMorita/polyhedron_distortion/Tutorial1_API.ipynb)

# Citation
See the following paper for the theoretical background.
(link to the publication to be added)

