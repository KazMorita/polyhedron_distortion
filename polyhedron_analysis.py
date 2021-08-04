import numpy as np

import sys

# pymatgen==2022.0.8
import pymatgen
import pymatgen.io.vasp
import pymatgen.analysis.local_env
import pymatgen.analysis.molecule_matcher


list_irrep_distortions = [0.0, 0.5, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, -0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.4082482904638631, 0.0, 0.0, 0.4082482904638631, 0.0, 0.0, 0.4082482904638631, 0.0, 0.0, 0.4082482904638631, 0.0, 0.0, 0.4082482904638631, 0.0, 0.0, 0.4082482904638631, 0.0, 0.0, 0.0, 0.4082482904638631, 0.0, 0.0, 0.4082482904638631, 0.0, 0.0, 0.4082482904638631, 0.0, 0.0, 0.4082482904638631, 0.0, 0.0, 0.4082482904638631, 0.0, 0.0, 0.4082482904638631, 0.0, 0.0, 0.0, 0.4082482904638631, 0.0, 0.0, 0.4082482904638631, 0.0, 0.0, 0.4082482904638631, 0.0, 0.0, 0.4082482904638631, 0.0, 0.0, 0.4082482904638631, 0.0, 0.0, 0.4082482904638631, 0.4082482904638631, 0.0, 0.0, 0.0, 0.4082482904638631, 0.0, 0.0, 0.0, 0.4082482904638631, 0.0, 0.0, -0.4082482904638631, 0.0, -0.4082482904638631, 0.0, -0.4082482904638631, 0.0, 0.0, 0.5773502691896258, 0.0, 0.0, 0.0, -0.2886751345948129, 0.0, 0.0, 0.0, -0.2886751345948129, 0.0, 0.0, 0.2886751345948129, 0.0, 0.2886751345948129, 0.0, -0.5773502691896258, 0.0, 0.0, 7.401486830834377e-17, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.5, 0.0, -0.5, 0.0, -7.401486830834377e-17, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.4082482904638631, 0.0, 0.0, -0.4082482904638631, 0.0, 0.0, -0.4082482904638631, 0.0, 0.0, -0.4082482904638631, 0.0, 0.0, -0.4082482904638631, 0.0, 0.0, 0.4082482904638631, 0.0, 0.0, 0.0, -0.4082482904638631, 0.0, 0.0, 0.4082482904638631, 0.0, 0.0, -0.4082482904638631, 0.0, 0.0, -0.4082482904638631, 0.0, 0.0, 0.4082482904638631, 0.0, 0.0, -0.4082482904638631, 0.0, 0.0, 0.0, -0.4082482904638631, 0.0, 0.0, -0.4082482904638631, 0.0, 0.0, 0.4082482904638631, 0.0, 0.0, 0.4082482904638631, 0.0, 0.0, -0.4082482904638631, 0.0, 0.0, -0.4082482904638631, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, -0.5, 0.0, 0.0, -0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]


def calc_displacement(struct, centre_atom, nearest_neighbour_indices,
                           ave_bond, images):

    # create "molecules"
    pymatgen_molecule = pymatgen.core.structure.Molecule(
        species=[struct.sites[centre_atom]._species]
        + [struct.sites[i]._species for i in nearest_neighbour_indices],
        coords=np.concatenate((np.zeros((1, 3)), (np.array([(
            struct.sites[site]._coords
            + struct.lattice.get_cartesian_coords(images[i])
        )
            for i, site in enumerate(nearest_neighbour_indices)
        ]) - struct.sites[centre_atom]._coords
        ) / ave_bond))
    )
    pymatgen_molecule_ideal = pymatgen.core.structure.Molecule(
        species=pymatgen_molecule.species,
        coords=np.concatenate((np.zeros((1, 3)), [
            [-1,  0,  0],
            [0, -1,  0],
            [0,  0, -1],
            [0,  0,  1],
            [0,  1,  0],
            [1,  0,  0]
        ])))

    # match "molecular" structure (Hungarian algorithm)
    (inds, u, v, _) = pymatgen.analysis.molecule_matcher.HungarianOrderMatcher(
        pymatgen_molecule_ideal).match(pymatgen_molecule)

    # affine transform
    pymatgen_molecule.apply_operation(pymatgen.core.operations.SymmOp(
            np.concatenate((
                np.concatenate((u.T, v.reshape(3, 1)), axis=1),
                [np.zeros(4)]), axis=0
            )))
    pymatgen_molecule._sites = np.array(
        pymatgen_molecule._sites)[inds].tolist()

    # project
    distortion_amplitudes = np.tensordot(
        np.array(list_irrep_distortions).reshape(18, 18),
        (pymatgen_molecule.cart_coords
         - pymatgen_molecule_ideal.cart_coords).ravel()[3:], axes=1)
    distortion_amplitudes = distortion_amplitudes * distortion_amplitudes
    distortion_amplitudes = np.array([
        distortion_amplitudes[7] + distortion_amplitudes[8],
        distortion_amplitudes[9] + distortion_amplitudes[10] + distortion_amplitudes[11],
        distortion_amplitudes[12] + distortion_amplitudes[13] + distortion_amplitudes[14],
        distortion_amplitudes[15] + distortion_amplitudes[16] + distortion_amplitudes[17],
    ])
    return np.sqrt(distortion_amplitudes)


def calc_distortions_from_struct(mp_struct, centre_atom):
    # handle nearest neighbours
    mp_struct.get_neighbor_list(r=3.5)
    nearest_neighbour_finder = pymatgen.analysis.local_env.CrystalNN()
    temp_dict = sorted(
        nearest_neighbour_finder.get_nn_info(
            structure=mp_struct, n=centre_atom), key=lambda x: -x['weight']
    )[:6]

    return (calc_displacement(
        struct=mp_struct,
        centre_atom=centre_atom,
        nearest_neighbour_indices=[d['site_index'] for d in temp_dict],
        ave_bond=np.mean([mp_struct.get_distance(
            centre_atom, d['site_index']) for d in temp_dict]),
        images=[d['image'] for d in temp_dict])
    )


def main(): # for vasp input
    # get arguments
    argvs = sys.argv
    INFILE   = argvs[1]  # POSCAR
    centre_atom = int(argvs[2]) - 1

    # convert to pymatgen
    mp_struct = pymatgen.io.vasp.inputs.Poscar.from_file(INFILE).structure

    # main analysis
    print('#Eg, T2g, T1u, T2u')
    print(calc_distortions_from_struct(mp_struct, centre_atom))

    return 0


if __name__ == '__main__':
    sys.exit(main())

