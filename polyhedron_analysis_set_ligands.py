#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import sys

import numpy as np

# pymatgen==2022.0.8
import pymatgen
import pymatgen.io.vasp
import pymatgen.analysis.local_env
import pymatgen.analysis.molecule_matcher

from polyhedron_analysis import construct_molecule, construct_molecule_ideal, match_molecules, calc_displacement, calc_displacement_centre


def calc_distortions_from_struct_octahedron_withcentre_withligands(
        mp_struct, centre_atom, ligand_atoms):
    # check args
    if(len(ligand_atoms) != 6):
        sys.stderr.write('Error: len(ligand_atoms) must be 6.\n')
        sys.exit(0)

    # constants
    ideal_coords = [
        [-1,  0,  0],
        [0, -1,  0],
        [0,  0, -1],
        [0,  0,  1],
        [0,  1,  0],
        [1,  0,  0]
    ]
    filename_basis = (
        os.path.dirname(os.path.realpath(__file__))
        + '/basis/octahedron_basis.json'
    )

    # read json
    try:
        with open(filename_basis, 'r') as f:
            dict_basis = json.load(f)
    except IOError:
        sys.stderr.write('IOError: failed reading from {}.'
                         .format(filename_basis))
        sys.exit(1)
    irrep_distortions = []
    for irrep in dict_basis.keys():
        for elem in dict_basis[irrep]:
            irrep_distortions.append(elem)

    # handle nearest neighbours
    temp_dict_all_sites = mp_struct.get_all_neighbors(
        r=5.0, sites=[mp_struct.sites[centre_atom]], numerical_tol=1e-08)
    temp_dict = [
        {
            'site': d,
            'image': d.image,
            'site_index': d.index,
        } for d in temp_dict_all_sites[0] if d.index in ligand_atoms
    ]
    if(len(temp_dict) < 6):
        sys.stderr.write('failed to find nearest neighbours\n')
        sys.exit(1)

    # define "molecules"
    temp_coords = np.array([
        mp_struct[d['site_index']].coords
        + mp_struct.lattice.get_cartesian_coords(d['image'])
        for d in temp_dict
    ])
    molecule_origin = np.mean(temp_coords, axis=0)
    ave_bond = temp_coords - molecule_origin
    ave_bond = np.mean(np.sqrt(np.sum(ave_bond * ave_bond, axis=1)))

    pymatgen_molecule = construct_molecule(
        struct=mp_struct,
        centre_atom=centre_atom,
        nearest_neighbour_indices=[d['site_index'] for d in temp_dict],
        ave_bond=ave_bond,
        images=[d['image'] for d in temp_dict],
        origin=molecule_origin,
    )
    pymatgen_molecule_ideal = construct_molecule_ideal(
        ideal_coords, pymatgen_molecule.species)

    # transform
    (pymatgen_molecule, matrix_rotation, _) = match_molecules(
        pymatgen_molecule, pymatgen_molecule_ideal)

    # project
    distortion_amplitudes = calc_displacement(
        pymatgen_molecule, pymatgen_molecule_ideal, irrep_distortions
    )

    # calc projection for central atom
    centre_atom_amplitude = calc_displacement_centre(
        mp_struct, centre_atom, molecule_origin, ave_bond, matrix_rotation)
    centre_atom_amplitude = np.sqrt(
        np.sum(centre_atom_amplitude * centre_atom_amplitude))

    # average
    distortion_amplitudes = distortion_amplitudes * distortion_amplitudes
    temp_list = []
    count = 0
    for irrep in dict_basis:
        dim = len(dict_basis[irrep])
        temp_list.append(np.sum(distortion_amplitudes[count:count + dim]))
        count += dim
    distortion_amplitudes = np.sqrt(temp_list)[3:]

    return np.concatenate((distortion_amplitudes, [centre_atom_amplitude]))


def main():  # for vasp input
    # get arguments
    argvs = sys.argv
    INFILE = argvs[1]  # POSCAR
    centre_atom = int(argvs[2]) - 1
    ligand_atoms = [int(argvs[3]) - 1,
                    int(argvs[4]) - 1,
                    int(argvs[5]) - 1,
                    int(argvs[6]) - 1,
                    int(argvs[7]) - 1,
                    int(argvs[8]) - 1,
                    ]

    # convert to pymatgen
    mp_struct = pymatgen.io.vasp.inputs.Poscar.from_file(
        INFILE, check_for_POTCAR=False).structure

    # main analysis
    print('#Eg, T2g, T1u, T2u, T1u(centre)')
    print(calc_distortions_from_struct_octahedron_withcentre_withligands(
        mp_struct, centre_atom, ligand_atoms
    ))

    return 0


if __name__ == '__main__':
    sys.exit(main())
