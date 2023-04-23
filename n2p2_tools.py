import numpy as np
import pynnp
from pathlib import Path


atoms_dict = {
    'H': 1, 1: 'H',
    'C': 6, 6: 'C',
    'N': 7, 7: 'N',
    'O': 8, 8: 'O',
    'F': 9, 9: 'F',
    'S': 16, 16: 'S'
}


class CNN:
    def __init__(self, top_folder, input_nn=None, scaling_data=None, n_members=1, stats_file=None):
        """
        Args:
            input_nn: str
                Path to input.nn file containing settings of NN, shared by all committee members.
            scaling_data: str
                Path to scaling.data file, shared by all members.
            top_folder:
                Path to the folder containing members (usually nnp-??) sub-folders which contain weights.?????.data.
            n_members: int
                Path to the input.data file.
        """

        self.input_nn = input_nn
        self.scaling_data = scaling_data
        self.top_folder = top_folder
        self.n_members = n_members
        self.models = None
        self.setup_models()
        self.stats_file = stats_file

    def setup_models(self):
        self.get_settings()
        self.get_member_dirs()
        self.models = []
        for member_dir in self.member_dirs:
            m_temp = pynnp.Mode()
            m_temp.initialize()
            m_temp.loadSettingsFile(self.input_nn)
            m_temp.setupNormalization()
            m_temp.setupElementMap()
            m_temp.setupElements()
            m_temp.setupCutoff()
            m_temp.setupSymmetryFunctions()
            m_temp.setupSymmetryFunctionMemory()  # comment out when compiled with N2P2_FULL_SFD_MEMORY
            m_temp.setupSymmetryFunctionCache()  # comment out when compiled with N2P2_NO_SF_CACHE
            m_temp.setupSymmetryFunctionGroups()
            m_temp.setupNeuralNetwork()
            m_temp.setupSymmetryFunctionScaling(self.scaling_data)
            m_temp.setupSymmetryFunctionStatistics(
                collectStatistics=False,
                collectExtrapolationWarnings=True,
                writeExtrapolationWarnings=True,
                stopOnExtrapolationWarnings=False
            )

            m_temp.setupNeuralNetworkWeights(f'{member_dir}/weights.%03zu.data')

            self.models.append(m_temp)

    def get_member_dirs(self):
        all_weights_paths = Path(self.top_folder).glob('*/weights.*.data')
        member_dirs = set()
        for weight_path in all_weights_paths:
            member_dirs.add(weight_path.parent)
        self.member_dirs = list(member_dirs)
        self.n_members = len(self.member_dirs)
        return self.member_dirs

    def get_settings(self):
        self.input_nn = str(list(Path(self.top_folder).glob('*/input.nn'))[0])
        self.scaling_data = str(list(Path(self.top_folder).glob('*/scaling.data'))[0])


def ase_atoms_to_n2p2(atoms, elementmap=None):
    """
    A tool to convert ASE Atoms object to pynnp.Structure object.
    Args:
        atoms: ASE Atoms object
            input coordinates.
        elementmap: pynnp.ElementMap object
            An elementmap is required to add atoms to the structure.
            elementmap can be set up in a model = pynnp.Mode() with model.setupElementMap()

    Returns:
        strucutre: pynnp.Structure

    """
    atoms_pos = atoms.positions
    elements = [atoms_dict[atomic_number] for atomic_number in atoms.numbers]

    # converting to Custom n2p2 format
    structure = pynnp.Structure()
    structure.setElementMap(elementmap)
    for i, pos in enumerate(atoms_pos):
        x, y, z = pos
        temp_atom = pynnp.Atom()
        temp_atom.r = pynnp.Vec3D(x=x, y=y, z=z)
        structure.addAtom(temp_atom, elements[i])

    structure.isPeriodic = atoms.pbc.any()
    structure.pbc = [int(atoms.pbc[0]), int(atoms.pbc[1]), int(atoms.pbc[2])]
    return structure


def n2p2_committee_calculator(cnn, structure):
    """
    Takes a list of models (committee) and return average electronic coupling and standard deviation of predictions.
    Args:
        cnn: CNN object
            Committee neural network configuration class.
        structure: pynnp.Structure
            Path to the input.data file.

    Returns:
        electronic_coupling: float
            Average electronic coupling values predicted by committee of NNPs.
        std_of_ec: float
            Standard devation of predictions of committee members.
    """

    committee_ec = []
    for i_m, m in enumerate(cnn.models):
        m.removeEnergyOffset(structure)
        # If normalization is used, convert structure data.
        if m.useNormalization():
            structure.toNormalizedUnits(
                m.getMeanEnergy(),
                m.getConvEnergy(),
                m.getConvLength()
            )

        if i_m == 0:
            # Retrieve cutoff radius form NNP setup.
            cutoffRadius = m.getMaxCutoffRadius()
            # print("Cutoff radius = ", cutoffRadius / convLength)

            # Calculate neighbor list.
            structure.calculateNeighborList(cutoffRadius)

            # Calculate symmetry functions for all atoms (use groups).
            # m.calculateSymmetryFunctions(s, False)
            m.calculateSymmetryFunctionGroups(structure, derivatives=True)

        m.calculateAtomicNeuralNetworks(structure, True)

        # Sum up potential energy.
        m.calculateEnergy(structure)

        # If normalization is used, convert structure data back to physical units.
        if m.useNormalization():
            structure.toPhysicalUnits(
                m.getMeanEnergy(),
                m.getConvEnergy(),
                m.getConvLength()
            )
        m.addEnergyOffset(structure, ref=False)
        m.addEnergyOffset(structure, ref=True)

        committee_ec.append(structure.energy)

    electronic_coupling = np.mean(committee_ec)
    std_of_ec = np.std(committee_ec)

    return electronic_coupling, std_of_ec

