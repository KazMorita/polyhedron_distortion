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


def construct_molecule(struct, centre_atom, nearest_neighbour_indices,
                       ave_bond, images):
    pymatgen_molecule = pymatgen.core.structure.Molecule(
        species=[struct.sites[centre_atom]._species]
        + [struct.sites[i]._species for i in nearest_neighbour_indices],
        coords=np.concatenate((np.zeros((1, 3)), (np.array([(
            struct.sites[site].coords
            + struct.lattice.get_cartesian_coords(images[i])
        )
            for i, site in enumerate(nearest_neighbour_indices)
        ]) - struct.sites[centre_atom].coords
        ) / ave_bond), axis=0)
    )
    return pymatgen_molecule


def construct_molecule_ideal(ideal_coords, species):
    return pymatgen.core.structure.Molecule(
        species=species,
        coords=np.concatenate((np.zeros((1, 3)), ideal_coords), axis=0))


def match_molecules(molecule_transform, molecule_reference):
    # match molecules
    (inds, u, v, _) = pymatgen.analysis.molecule_matcher.HungarianOrderMatcher(
        molecule_reference).match(molecule_transform)

    # affine transform
    molecule_transform.apply_operation(pymatgen.core.operations.SymmOp(
        np.concatenate((
            np.concatenate((u.T, v.reshape(3, 1)), axis=1),
            [np.zeros(4)]), axis=0
        )))
    molecule_transform._sites = np.array(
        molecule_transform._sites)[inds].tolist()

    return molecule_transform


def calc_displacement(
        pymatgen_molecule, pymatgen_molecule_ideal, irrep_distortions):
    return np.tensordot(
        irrep_distortions,
        (pymatgen_molecule.cart_coords
         - pymatgen_molecule_ideal.cart_coords).ravel()[3:], axes=1)


def calc_distortions_from_struct_octahedron(mp_struct, centre_atom):
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
    mp_struct.get_neighbor_list(r=3.5)
    nearest_neighbour_finder = pymatgen.analysis.local_env.CrystalNN()
    temp_dict = sorted(
        nearest_neighbour_finder.get_nn_info(
            structure=mp_struct, n=centre_atom), key=lambda x: -x['weight']
    )[:len(ideal_coords)]

    # define "molecules"
    pymatgen_molecule = construct_molecule(
        struct=mp_struct,
        centre_atom=centre_atom,
        nearest_neighbour_indices=[d['site_index'] for d in temp_dict],
        ave_bond=np.mean([
            mp_struct.get_distance(centre_atom, d['site_index'],
                                   d['image']) for d in temp_dict
        ]),
        images=[d['image'] for d in temp_dict],
    )
    pymatgen_molecule_ideal = construct_molecule_ideal(
        ideal_coords, pymatgen_molecule.species)

    # transform
    pymatgen_molecule = match_molecules(
        pymatgen_molecule, pymatgen_molecule_ideal)

    # project
    distortion_amplitudes = calc_displacement(
        pymatgen_molecule, pymatgen_molecule_ideal, irrep_distortions
    )

    # average
    distortion_amplitudes = distortion_amplitudes * distortion_amplitudes
    temp_list = []
    count = 0
    for irrep in dict_basis:
        dim = len(dict_basis[irrep])
        temp_list.append(np.sum(distortion_amplitudes[count:count + dim]))
        count += dim
    distortion_amplitudes = np.sqrt(temp_list)[3:]

    return distortion_amplitudes


def main():  # for vasp input
    # get arguments
    argvs = sys.argv
    INFILE = argvs[1]  # POSCAR
    centre_atom = int(argvs[2]) - 1

    # convert to pymatgen
    mp_struct = pymatgen.io.vasp.inputs.Poscar.from_file(INFILE).structure

    # main analysis
    print('#Eg, T2g, T1u, T2u')
    print(calc_distortions_from_struct_octahedron(mp_struct, centre_atom))

    return 0


if __name__ == '__main__':
    sys.exit(main())
