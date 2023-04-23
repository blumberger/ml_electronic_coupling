from src.pyAOM_utils import *
from n2p2_tools import CNN, ase_atoms_to_n2p2, n2p2_committee_calculator
from ase.io import read


def aom(xyz_fname, projection_coeffs, MO, C):
    """
    Takes xyz coordinates of dimer atoms, projection coefficients, and C_aom and return electronic coupling.
    Args:
        xyz_fname: str
            Name of XYZ file containing atomic coordinates of dimer atoms.
        projection_coeffs: str
            Name of file containing projection coefficients
        MO: int
            FMO index
        C: float
            C_aom
    Returns:
        electronic_coupling: float
    """

    # STO decay coefficienrs
    AOM_dict={
        'H': {'STOs': 1, '1s': 1.0000},
        'C': {'STOs': 1+3, '2s': 1.6083, '2p': 1.385600},
        'N': {'STOs': 1+3, '2s': 1.9237, '2p': 1.617102},
        'O': {'STOs': 1+3, '2s': 2.2458, '2p': 1.505135},
        'F': {'STOs': 1+3, '2s': 2.5638, '2p': 1.665190},
        'S': {'STOs': 1+3, '3s': 2.1223, '3p': 1.641119},
    }

    # define AOM project
    frag1_AOM_file = projection_coeffs
    frag2_AOM_file = projection_coeffs

    # define MOs
    frag1_MO = MO
    frag2_MO = MO

    overlap = Sab(xyz_fname, frag1_AOM_file, frag2_AOM_file, frag1_MO, frag2_MO, AOM_dict)
    el_coupling = C * overlap

    return el_coupling


def neural_net_prediction(xyz_fname, cnn):
    """
    Takes a committee neural network object and coordinates of dimers atoms and returns electronic coupling
    Args:
        xyz_fname: str
            Name of XYZ file containing atomic coordinates of dimer atoms.
        cnn: CNN object.
            A CNN object that contains neural networks of members of committee.
    Returns:
        electronic_coupling: float
    """
    atoms = read(xyz_fname)
    structure = ase_atoms_to_n2p2(atoms, elementmap=cnn.models[0].elementMap)
    electronic_coupling, std_of_ec = n2p2_committee_calculator(
        cnn,
        structure,
    )
    return electronic_coupling, std_of_ec


def delta_ml(xyz_fname, projection_coeffs, MO, C, cnn):
    """
    Takes AOM and ML correction parameters and returns ML-corrected electronic couplings
    Args:
        xyz_fname: str
            Name of XYZ file containing atomic coordinates of dimer atoms.
        projection_coeffs: str
            Name of file containing projection coefficients
        MO: int
            FMO index
        C: float
            C_aom
        cnn: CNN object.
            A CNN object that contains neural networks of members of committee.

    Returns:
        electronic_coupling: float
    """
    ecp_aom_ = aom(xyz_fname, projection_coeffs, MO, C)
    ecp_cnn_, std_ = neural_net_prediction(xyz_fname, cnn)

    return ecp_aom_ + ecp_cnn_


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description=''
    )

    parser.add_argument(
        "files",
        type=str,
        nargs='+',
        help='xyz files containing coordinates of dimers atoms.',
        default=False
    )

    parser.add_argument(
        "--mode",
        type=str,
        help="Mode of calculation. Choices: 'aom' for AOM values only, 'nn' for direct neural network prediction, and"
             "'delta_ml' for AOM + nn correction ",
        required=True
    )

    parser.add_argument(
        "--cnn_dir",
        type=str,
        help='Top folder that contains the network parameters, scaling information, weights, and biases of members'
             'of the committee.'
    )

    parser.add_argument(
        "-c", "--c_aom",
        type=float,
        help='C parameter in AOM model.'
    )

    parser.add_argument(
        "--mo",
        type=int,
        help='Index of frontier molecular orbital (FMO) for AOM predictions.'
    )

    parser.add_argument(
        "-p", "--proj_coeffs",
        type=str,
        help='Name of file containing projection coefficients for AOM model.'
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        help='Name of file to which outputs are written.',
        default='electronic_couplings.dat'
    )

    args = parser.parse_args()

    if args.mode == 'nn' or args.mode == 'delta_ml':
        cnn = CNN(top_folder=args.cnn_dir)

    with open(args.output, 'w') as f:
        f.write(f'# filename  -- {args.mode} coupling\n')
        for file in args.files:
            if args.mode == 'aom':
                ec = aom(file, args.proj_coeffs, args.mo, args.c_aom)
            elif args.mode == 'nn':
                ec, std = neural_net_prediction(file, cnn)
            elif args.mode == 'delta_ml':
                ec = delta_ml(file, args.proj_coeffs, args.mo, args.c_aom, cnn)
            else:
                raise Exception(f'{args.mode} mode is not available. Valid choices are: aom, nn, and delta_ml')
            f.write(f'{file}  {ec:>12.6f}\n')


if __name__ == '__main__':
    main()

