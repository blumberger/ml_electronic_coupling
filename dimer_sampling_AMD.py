import numpy as np
from scipy.spatial import distance_matrix
from chainer import cuda


atoms_dict = {
    'H': 1, 1: 'H',
    'C': 6, 6: 'C',
    'N': 7, 7: 'N',
    'O': 8, 8: 'O'
}


class Atom:
    def __init__(self, element, position, force, charge):
        self.element = element
        self.position = position
        self.force = force
        self.charge = charge

    @property
    def atomic_number(self):
        return atoms_dict[self.element]


def xyz_reader(fname, has_header=False):

    atoms_list = []
    with open(fname, 'r') as g:
        lines = g.readlines()

        if has_header:
            lines = lines[2:]

        for line in lines:
            parts = line.split()
            if len(parts) < 4:
                continue
            atoms_list.append(Atom(parts[0],
                                   [float(parts[1]), float(parts[2]), float(parts[3])],
                                   [0, 0, 0],
                                   0))
    return atoms_list


def AMD_dimers(xyz_fname, has_header=False):

    atoms_list = xyz_reader(xyz_fname, has_header)

    atoms_positions = []
    for atom in atoms_list:
        atoms_positions.append(atom.position)
    atoms_positions = np.array(atoms_positions)

    distance_mat_ = distance_matrix(atoms_positions, atoms_positions)

    return np.mean(np.sort(distance_mat_, axis=1), axis=0)


def l2_norm(x, y):
    """Calculate l2 norm (distance) of `x` and `y`.
    Args:
        x (numpy.ndarray or cupy): (batch_size, num_point, coord_dim)
        y (numpy.ndarray): (batch_size, num_point, coord_dim)
    Returns (numpy.ndarray): (batch_size, num_point,)
    """
    return ((x - y) ** 2).sum(axis=2)


def farthest_point_sampling(pts, k, initial_idx=None, metrics=l2_norm,
                            skip_initial=False, indices_dtype=np.int32,
                            distances_dtype=np.float32):
    """Batch operation of farthest point sampling
    Code referenced from below link by @Graipher
    https://codereview.stackexchange.com/questions/179561/farthest-point-algorithm-in-python
    Args:
        pts (numpy.ndarray or cupy.ndarray): 2-dim array (num_point, coord_dim)
            or 3-dim array (batch_size, num_point, coord_dim)
            When input is 2-dim array, it is treated as 3-dim array with
            `batch_size=1`.
        k (int): number of points to sample
        initial_idx (int): initial index to start farthest point sampling.
            `None` indicates to sample from random index,
            in this case the returned value is not deterministic.
        metrics (callable): metrics function, indicates how to calc distance.
        skip_initial (bool): If True, initial point is skipped to store as
            farthest point. It stabilizes the function output.
        xp (numpy or cupy):
        indices_dtype (): dtype of output `indices`
        distances_dtype (): dtype of output `distances`
    Returns (tuple): `indices` and `distances`.
        indices (numpy.ndarray or cupy.ndarray): 2-dim array (batch_size, k, )
            indices of sampled farthest points.
            `pts[indices[i, j]]` represents `i-th` batch element of `j-th`
            farthest point.
        distances (numpy.ndarray or cupy.ndarray): 3-dim array
            (batch_size, k, num_point)
    """
    if pts.ndim == 2:
        # insert batch_size axis
        pts = pts[None, ...]
    assert pts.ndim == 3
    xp = cuda.get_array_module(pts)
    batch_size, num_point, coord_dim = pts.shape
    indices = xp.zeros((batch_size, k, ), dtype=indices_dtype)

    # distances[bs, i, j] is distance between i-th farthest point `pts[bs, i]`
    # and j-th input point `pts[bs, j]`.
    distances = xp.zeros((batch_size, k, num_point), dtype=distances_dtype)
    if initial_idx is None:
        indices[:, 0] = xp.random.randint(len(pts))
    else:
        indices[:, 0] = initial_idx

    batch_indices = xp.arange(batch_size)
    farthest_point = pts[batch_indices, indices[:, 0]]
    # minimum distances to the sampled farthest point
    try:
        min_distances = metrics(farthest_point[:, None, :], pts)
    except Exception as e:
        import IPython; IPython.embed()

    if skip_initial:
        # Override 0-th `indices` by the farthest point of `initial_idx`
        indices[:, 0] = xp.argmax(min_distances, axis=1)
        farthest_point = pts[batch_indices, indices[:, 0]]
        min_distances = metrics(farthest_point[:, None, :], pts)

    distances[:, 0, :] = min_distances
    for i in range(1, k):
        indices[:, i] = xp.argmax(min_distances, axis=1)
        farthest_point = pts[batch_indices, indices[:, i]]
        dist = metrics(farthest_point[:, None, :], pts)
        distances[:, i, :] = dist
        min_distances = xp.minimum(min_distances, dist)
    return indices, distances


if __name__ == '__main__':
    from pathlib import Path
    import argparse

    parser = argparse.ArgumentParser(
        description='Takes xyz files and returns the most diverse structures')

    parser.add_argument('-oa', '--picked_output',
                        metavar='',
                        help='Output file for list of picked IDs.',
                        default = 'IDs_picked.txt'
                        )

    parser.add_argument('-or', '--rejected_output',
                        metavar='',
                        help='Output file for list of not picked IDs.',
                        default='IDs_rejected.txt'
                        )

    parser.add_argument('-d', '--XYZfolder',
                        metavar='',
                        help='The folder that contains XYZ files of dimeers',
                        required=True
                        )

    parser.add_argument('-n', '--npoints',
                        metavar='',
                        help='The number of points needs to be sampled.',
                        default=50
                        )

    parser.add_argument('-wh', '--with_xyz_header',
                        help='Use it for when the XYZ file has header.',
                        dest='has_header',
                        action='store_true')
    parser.set_defaults(has_header=False)

    args = parser.parse_args()

    amd_dict = {}
    xyz_files = Path(args.XYZfolder).glob('*.xyz')

    for xyz_file in xyz_files:
        amd_dict[xyz_file.name] = AMD_dimers(xyz_file, has_header = args.has_header)

    amd_list = []
    keys_list = []
    for key in sorted(amd_dict.keys()):
        amd_list.append(amd_dict[key])
        keys_list.append(key)
    amd_list = np.array(amd_list)

    distance_mat = distance_matrix(amd_list, amd_list)

    neighbours, _ = farthest_point_sampling(amd_list, int(args.npoints))

    picked_keys = []
    with open(args.picked_output, 'w') as f:
        for neigh in neighbours[0]:
            picked_keys.append(keys_list[neigh])
            f.write(f'{keys_list[neigh]}\n')

    with open(args.rejected_output, 'w') as f:
        for key in keys_list:
            if key in picked_keys:
                continue
            f.write(f'{key}\n')
