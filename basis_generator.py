#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
import json
import os
import sys

import numpy as np
import phonopy  # version 2.8

THRESHOLD = 1e-12


def separate_translation(list_basis):
    translation_vec = np.zeros([3, len(list_basis[0])])
    translation_vec[0, 0::3] = 1.0 / (len(list_basis[0]) / 3.0)
    translation_vec[1, 1::3] = 1.0 / (len(list_basis[0]) / 3.0)
    translation_vec[2, 2::3] = 1.0 / (len(list_basis[0]) / 3.0)

    mat_transform = np.zeros([len(list_basis), len(list_basis)])

    list_basis_normalised = normalise_list(list_basis)

    mat_transform[:3] = np.tensordot(
        translation_vec[:3], list_basis_normalised, axes=[1, 1])

    for i in range(3, len(list_basis)):
        temp_row = np.zeros(len(list_basis))
        temp_row[i] = 1.0
        mat_transform[i] = calc_residual_GramSchmidt(
            mat_transform[:i], [temp_row])[0]
    mat_transform = normalise_list(mat_transform)

    # apply transform
    return np.dot(mat_transform, list_basis_normalised)


def separate_rotation(list_basis, coords):
    rotation_vec = np.zeros([3, len(list_basis[0])])
    mat_transform = np.zeros([len(list_basis), len(list_basis)])

    list_basis_normalised = normalise_list(list_basis)

    for iatom, coord in enumerate(coords):
        for r in range(3):  # Rx, Ry, Rz
            temp_coords = np.array(coord)
            temp_coords[r] = 0.0
            rotation_vec[r, 3 * iatom: 3 * iatom + 3] = \
                np.cross(np.eye(3)[r], temp_coords)
    rotation_vec = normalise_list(rotation_vec)

    mat_transform[:3] = np.tensordot(
        rotation_vec[:3], list_basis_normalised, axes=[1, 1])

    for i in range(3, len(list_basis)):
        temp_row = np.zeros(len(list_basis))
        temp_row[i] = 1.0
        mat_transform[i] = calc_residual_GramSchmidt(
            mat_transform[:i], [temp_row])[0]
    mat_transform = normalise_list(mat_transform)
    return np.dot(mat_transform, list_basis_normalised)


def normalise_list(list_vec):
    return (np.array(list_vec) / np.linalg.norm(list_vec, axis=1).reshape(-1, 1)).tolist()


def check_orthogonality_in_dict(dict_basis):
    temp_list = []
    for i in dict_basis.keys():
        for basis in dict_basis[i]:
            temp_list.append(basis)
    for (i, j) in itertools.combinations(temp_list, 2):
        prod = np.dot(i, j)
        if(np.fabs(prod) > THRESHOLD):
            sys.stderr.write('error not orthogonal\n')
            sys.exit(1)
    return 0


def check_orthogonality_in_list(target_vec, list_vec):
    for vec in list_vec:
        if(np.fabs(np.dot(target_vec, vec)) > THRESHOLD):
            return False  # not orthogonal
    return True


def get_vector_transformed_36(origin_atom_coor, transformed_atom_coor,
                              ref_vector, transformation_matrix, class_coef):
    return ((np.matrix(origin_atom_coor + ref_vector) * np.matrix(
        transformation_matrix)).A1 - transformed_atom_coor) * class_coef


def calc_residual_GramSchmidt(list_ortho_vec, list_target_vec):
    ret_list_vec = []
    for vec in list_target_vec:
        temp_vec = vec
        for basis in list_ortho_vec:
            basis_normalised = basis / np.linalg.norm(basis)
            temp_vec -= np.dot(temp_vec, basis_normalised) * basis_normalised
        ret_list_vec.append(temp_vec)
    return ret_list_vec


def basis_generating_machine_character(coords, point_group, basis_sorter):
    character_table = phonopy.phonon.irreps.character_table[point_group][0]
    list_reference_atoms = []
    reference_vectors = []
    reference_vector = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0])
    ]

    for reference_atom in range(len(coords)):
        for reference_direction in reference_vector:
            temp_vec = np.zeros(3 * len(coords))
            temp_vec[3 * reference_atom: 3
                     * reference_atom + 3] = reference_direction
            reference_vectors.append(temp_vec)
            list_reference_atoms.append(reference_atom)

    # initialise
    character_table['operator_onehot_atomic_redrep'] = {}
    for c_name in character_table['character_table']:
        character_table['operator_onehot_atomic_redrep'][c_name] = (
            np.zeros((len(reference_vectors), 3
                      * len(coords)))  # xyz
        )

    # calculate representation in atomic index basis
    character_table['mapping_table_atom_basis'] = {}
    for i_tname, t_name in enumerate(character_table['rotation_list']):
        temp_list_representations = []
        for t_namei in character_table['mapping_table'][t_name]:
            temp_representation = np.zeros([len(coords), len(coords)])
            for i, r_i in enumerate(coords):
                # for coordinates
                r_i_t = (np.matrix(r_i) * np.matrix(t_namei)).A1
                for j, r_j in enumerate(coords):
                    if(np.linalg.norm(r_j - r_i_t) < THRESHOLD):
                        temp_representation[i][j] = 1
                        break
                    if(j == len(coords)):
                        sys.stderr.write('representation not found')
                        sys.exit(1)
            temp_list_representations.append(temp_representation)
        character_table['mapping_table_atom_basis'][t_name] = \
            temp_list_representations

    # basis set generating machine in character convention
    for ir, reference_vector in enumerate(reference_vectors):
        for (c_name, i_tname) in itertools.product(
            character_table['character_table'],
            range(len(character_table['rotation_list']))
        ):
            t_name = character_table['rotation_list'][i_tname]

            for i_tnamei, t_namei in enumerate(
                    character_table['mapping_table_atom_basis'][t_name]):
                i_atom_trans = np.argmax(t_namei[list_reference_atoms[ir]])
                character_table['operator_onehot_atomic_redrep'][c_name][ir][
                    3 * i_atom_trans:3 * i_atom_trans + 3
                ] += get_vector_transformed_36(
                    coords[list_reference_atoms[ir]],
                    coords[i_atom_trans],
                    reference_vector[3 * list_reference_atoms[ir]:
                                     3 * list_reference_atoms[ir] + 3],
                    character_table['mapping_table'][t_name][i_tnamei],
                    character_table['character_table'][c_name][i_tname]
                )

    # delete ones that are not orthogonal ( or equal )
    character_table['operator_onehot_atomic_redrep_orthogonal'] = {}
    for c_name in character_table['character_table']:
        temp_operator_onehot_atomic_redrep = []
        for i_op, op in enumerate(
                character_table['operator_onehot_atomic_redrep'][c_name]):
            if(np.fabs(np.linalg.norm(op)) < THRESHOLD):
                continue
            elif(check_orthogonality_in_list(op, temp_operator_onehot_atomic_redrep)):
                temp_operator_onehot_atomic_redrep.append(op)
            else:  # if not orthogonal try to make it orthogonal
                residual = calc_residual_GramSchmidt(
                    temp_operator_onehot_atomic_redrep, [op])[0]
                if(np.fabs(np.linalg.norm(residual)) > THRESHOLD):
                    temp_operator_onehot_atomic_redrep.append(residual)
        character_table['operator_onehot_atomic_redrep_orthogonal'][c_name] \
            = temp_operator_onehot_atomic_redrep

    # post-process
    normal_basis = basis_sorter(
        character_table['operator_onehot_atomic_redrep_orthogonal'])

    for key in normal_basis.keys():
        normal_basis[key] = normalise_list(np.array(normal_basis[key]))
    check_orthogonality_in_dict(normal_basis)

    return normal_basis


# may ill behave for python <3.6
def sort_basis_numerical(
        dict_basis, translation_irrep, rotation_irrep, ideal_coords):
    trans = separate_translation(dict_basis[translation_irrep])
    rot = separate_rotation(dict_basis[rotation_irrep], ideal_coords)
    ret_dict = {
        'rotation' + rotation_irrep: rot[:3],
        'translation' + translation_irrep: trans[:3],
    }

    for irrep in dict_basis.keys():
        if(dict_basis[irrep] != []):
            if(irrep not in [translation_irrep, rotation_irrep]):
                ret_dict[irrep] = dict_basis[irrep]
            elif(irrep == translation_irrep and len(trans) > 3):
                ret_dict[irrep] = trans[3:]
            elif(irrep == rotation_irrep and len(rot) > 3):
                ret_dict[irrep] = rot[3:]
    return ret_dict


def sort_basis_analytical_octahderon(dict_basis):
    dict_basis['T1u'] = normalise_list(dict_basis['T1u'])
    return {
        'rotationT1g': dict_basis['T1g'][:3],
        'translationT1u': [
            (np.array(dict_basis['T1u'][0])
                + np.sqrt(2.0) * np.array(dict_basis['T1u'][3])).tolist(),
            (np.array(dict_basis['T1u'][4])
                + np.sqrt(2.0) * np.array(dict_basis['T1u'][1])).tolist(),
            (np.array(dict_basis['T1u'][5])
                + np.sqrt(2.0) * np.array(dict_basis['T1u'][2])).tolist(),
        ],
        'A1g': dict_basis['A1g'],
        'Eg': dict_basis['Eg'],
        'T2g': dict_basis['T2g'],
        'T1u': [
            (- np.sqrt(2.0) * np.array(dict_basis['T1u'][0])
                + np.array(dict_basis['T1u'][3])).tolist(),
            (- np.sqrt(2.0) * np.array(dict_basis['T1u'][4])
                + np.array(dict_basis['T1u'][1])).tolist(),
            (- np.sqrt(2.0) * np.array(dict_basis['T1u'][5])
                + np.array(dict_basis['T1u'][2])).tolist(),
        ],
        'T2u': dict_basis['T2u'],
    }


def main():
    # constants
    ideal_coords = [
        [-1,  0,  0],
        [0, -1,  0],
        [0,  0, -1],
        [0,  0,  1],
        [0,  1,  0],
        [1,  0,  0]
    ]

    # write to json (numerical solution)
    filename_basis = (
        os.path.dirname(os.path.realpath(__file__))
        + '/basis/octahedron_basis.json'
    )
    try:
        with open(filename_basis, 'w') as f:
            f.write(json.dumps(
                basis_generating_machine_character(
                    ideal_coords, point_group='m-3m',
                    basis_sorter=lambda x: sort_basis_numerical(
                        x, 'T1u', 'T1g', ideal_coords))
            ) + '\n')
    except IOError:
        sys.stderr.write(
            'IOError: failed writing to {}.'.format(filename_basis))
        sys.exit(1)

    # write to json (analytical solution)
    filename_basis = (
        os.path.dirname(os.path.realpath(__file__))
        + '/basis/octahedron_basis_analytical.json'
    )
    try:
        with open(filename_basis, 'w') as f:
            f.write(json.dumps(
                basis_generating_machine_character(
                    ideal_coords, point_group='m-3m',
                    basis_sorter=sort_basis_analytical_octahderon)
            ) + '\n')
    except IOError:
        sys.stderr.write(
            'IOError: failed writing to {}.'.format(filename_basis))
        sys.exit(1)

    return 0


if __name__ == '__main__':
    sys.exit(main())
