import numpy as np

from itertools import compress

from scipy import sparse
from scipy.ndimage import generate_binary_structure, label
from skan.csr import (
    make_degree_image,
    Skeleton,
    skeleton_to_csgraph,
    summarize,
)
from skimage.feature import blob_log
from skimage.graph import central_pixel, pixel_graph
from skimage.morphology import convex_hull_image
from skimage.segmentation import relabel_sequential
from skimage.util import invert


def _get_blobs(cell_image, image_type):
    """Extracts circular blobs in cell image for finding soma later.

    Parameters
    ----------
    cell_image : ndarray
        Grayscale image data of cell of nervous system.
    image_type : str
        Neuroimaging technique used to get image data of neuronal cell,
        either 'confocal' or 'DAB'.

    Returns
    -------
    ndarray
        Coordinates & radius of each blob.
    """

    if image_type == "DAB":
        min_sigma, max_sigma, num_sigma = 6, 20, 10
        threshold, overlap = 0.1, 0.5
        image = invert(cell_image)
    elif image_type == "confocal":
        min_sigma, max_sigma, num_sigma = 3, 20, 10
        threshold, overlap = 0.1, 0.5
        image = cell_image

    blobs_log = blob_log(image, min_sigma=min_sigma,
                         max_sigma=max_sigma, num_sigma=num_sigma,
                         threshold=threshold, overlap=overlap)

    def eliminate_border_blobs(blobs_log):
        """Find the blobs too close to border so as to eliminate them."""
        if image.ndim == 3:
            return blobs_log

        border_x = image.shape[-1] / 5
        border_y = image.shape[-2] / 5

        filtered_blobs = blobs_log[(border_x < blobs_log[:, 1]) &
                                   (blobs_log[:, 1] < 4*border_x) &
                                   (border_y < blobs_log[:, 0]) &
                                   (blobs_log[:, 0] < 4*border_y)]

        return filtered_blobs

    blobs_log = eliminate_border_blobs(blobs_log)

    if len(blobs_log) < 1:
        # if none of the blobs remain after border blob elimination,
        # try blob_log with less stringent parameters
        while len(blobs_log) < 1 and min_sigma > 2:  # TODO: handle fail
            min_sigma -= 1
            blobs_log = blob_log(image, min_sigma, max_sigma,
                                 num_sigma, threshold, overlap)
            blobs_log = eliminate_border_blobs(blobs_log)

    if len(blobs_log) < 1:
        raise RuntimeError('No blob detected for the soma!')

    return blobs_log


def _centre_of_mass(blobs, cell_image, image_type):
    """Finds centre of mass of the multiple blobs detected.

    Find the blob with highest intensity value.
    """
    ixs = np.indices(cell_image.shape)

    n_blobs = blobs.shape[0]
    blob_centres = blobs[:, 0:cell_image.ndim]
    blob_radii = blobs[:, cell_image.ndim]

    centres = blob_centres[..., np.newaxis, np.newaxis]
    radii = np.square(blob_radii)[:, np.newaxis, np.newaxis]
    if cell_image.ndim == 3:
        centres = centres[..., np.newaxis]
        radii = radii[..., np.newaxis]

    # Using the formula for a circle, `x**2 + y**2 < r**2`,
    # generate a mask for all blobs.
    mask = np.square(ixs - centres).sum(axis=1) < radii
    # Calculate the average intensity of pixels under the mask
    blob_intensities = np.full(
        (n_blobs, *ixs.shape[1:]), cell_image)
    blob_intensities = (blob_intensities * mask).reshape(mask.shape[0], -1)
    blob_intensities[blob_intensities < .02 * blob_intensities.max()] = 0
    blob_intensities = blob_intensities.sum(1) * (blob_intensities != 0).sum(1)

    if image_type == "DAB":
        max_intensity = blob_centres[np.argmin(blob_intensities)]

        return max_intensity

    elif image_type == "confocal":
        max_radius_idx = np.argmax(blob_radii)
        max_intensity_idx = np.argmax(blob_intensities)
        max_radius = blob_centres[max_radius_idx]
        max_intensity = blob_centres[max_intensity_idx]
        if np.count_nonzero(blob_radii == blob_radii[max_radius_idx]) > 1:
            return max_intensity
        return max_radius


def _get_soma(cell_image, image_type):
    """Calculate pixel position to be attribute as soma."""
    soma_blobs = _get_blobs(cell_image, image_type)

    if len(soma_blobs) == 0:
        raise RuntimeError('No soma detected for the cell!')
    if len(soma_blobs) == 1:
        soma = soma_blobs[0][:cell_image.ndim]
    if len(soma_blobs) > 1:
        soma = _centre_of_mass(soma_blobs, cell_image, image_type)

    return soma


def get_surface_area(im, scale):
    real_unit = np.prod(scale)
    npixels = np.sum(im)
    area = npixels * real_unit
    return area


def pad_skeleton(cell_skeleton, soma_on_skeleton):
    """Adds padding to cell skeleton image."""
    # get all the pixel indices representing skeleton
    skeleton_indices = np.nonzero(cell_skeleton)

    # get corner points enclosing skeleton
    if cell_skeleton.ndim == 2:
        x_min, x_max = min(skeleton_indices[1]), max(skeleton_indices[1]) + 1
        y_min, y_max = min(skeleton_indices[0]), max(skeleton_indices[0]) + 1
        bounded_skeleton = cell_skeleton[y_min:y_max, x_min:x_max]
    else:
        z_min, z_max = min(skeleton_indices[0]), max(skeleton_indices[0]) + 1
        y_min, y_max = min(skeleton_indices[1]), max(skeleton_indices[1]) + 1
        x_min, x_max = min(skeleton_indices[2]), max(skeleton_indices[2]) + 1
        bounded_skeleton = cell_skeleton[z_min:z_max, y_min:y_max, x_min:x_max]

    pad_width = max(bounded_skeleton.shape)//2
    nd_soma_on_skeleton = np.array(soma_on_skeleton)
    if cell_skeleton.ndim == 2:
        nd_min_bounds = np.array([y_min, x_min])
    else:
        nd_min_bounds = np.array([z_min, y_min, x_min])

    # get updated soma position on bounded and padded skeleton
    soma_on_bounded_skeleton = tuple(nd_soma_on_skeleton - nd_min_bounds)
    soma_on_padded_skeleton = tuple(
        nd_soma_on_skeleton - nd_min_bounds + pad_width)

    return (
        np.pad(bounded_skeleton, pad_width=pad_width, mode='constant'),
        soma_on_padded_skeleton,
        soma_on_bounded_skeleton
    )


def _eliminate_loops(summary, paths_list):
    """Eliminate loops in branches. handle all w/ type == 3"""
    cycle_branch_type = 3
    cycle_mask = summary['branch-type'] == cycle_branch_type
    cycle_branches_idx = summary[cycle_mask].index
    for i in cycle_branches_idx:
        paths_list[i] = paths_list[i][:-1]  # remove cycle node

    return paths_list


def _simplify_graph(skel):
    """Iterative removal of all nodes of degree 2 while reconnecting their
    edges.

    Parameters
    ----------
    skel : skan.csr.Skeleton
        A Skeleton object containing graph to be simplified.

    Returns
    -------
    simp_csgraph : scipy.sparse.csr_matrix
        A sparse adjacency matrix of the simplified graph.
    reduced_nodes : tuple of int
        The index nodes of original graph in simplified graph.
    """
    summary = summarize(skel)
    src = np.asarray(summary['node-id-src'])
    dst = np.asarray(summary['node-id-dst'])
    distance = np.asarray(summary['branch-distance'])

    # to reduce the size of simplified graph
    _, fw, inv = relabel_sequential(np.append(src, dst))
    src_relab, dst_relab = fw[src], fw[dst]

    n_nodes = max(np.max(src_relab), np.max(dst_relab))

    edges = sparse.coo_matrix(
            (distance, (src_relab - 1, dst_relab - 1)),
            shape=(n_nodes, n_nodes)
            )
    dir_csgraph = edges.tocsr()
    simp_csgraph = dir_csgraph + dir_csgraph.T  # make undirected

    reduced_nodes = inv[np.arange(1, simp_csgraph.shape[0] + 1)]

    return simp_csgraph, reduced_nodes


def _fast_graph_center_idx(skel):
    """Accelerated graph center finding using simplified graph.

    Parameters
    ----------
    skel : skan.csr.Skeleton
        A Skeleton object containing graph whose center is to be found.

    Returns
    -------
    original_center_idx : int
        The index of central node of graph.
    """
    simp_csgraph, reduced_nodes = _simplify_graph(skel)
    simp_center_idx, _ = central_pixel(simp_csgraph)
    original_center_idx = reduced_nodes[simp_center_idx]

    return original_center_idx


def get_soma_on_skeleton(cell):
    """Retrieves soma's position on cell skeleton."""
    # soma = _get_soma(cell_image, image_type)
    # if cell_image.ndim == 2:
    #     skeleton_pixel_coordinates = [(i, j) for (
    #         i, j), val in np.ndenumerate(cell_skeleton) if val != 0]
    # g, nodes = pixel_graph(cell_skeleton, connectivity=cell_image.ndim)
    # soma_on_skeleton, distances = central_pixel(
    #     g, nodes=nodes, shape=cell_image.shape, partition_size=100)
            # filter blobs with forks
    cell_image, image_type = cell.image, cell.image_type
    skel = cell._skeleton

    # soma finding
    original_center_idx = _fast_graph_center_idx(skel)

    neighbors = [original_center_idx]
    original_center_idx = neighbors.pop(0)

    itr = 0
    ITR_CUTOFF = 100
    while (
        cell_image[tuple(skel.coordinates[original_center_idx].astype(int))] == 0
        and itr < ITR_CUTOFF
    ):
        if len(neighbors) == 0:
            neighbors = list(skel.nbgraph.neighbors(original_center_idx))
        original_center_idx = neighbors.pop(0)
        itr += 1

    if itr < ITR_CUTOFF:
        soma_on_skeleton = skel.coordinates[original_center_idx].astype(int)
    else:
        print('Blob detection')
        soma = np.asarray(_get_soma(cell_image, image_type))
        if cell_image.ndim == 2:
            skeleton_pixel_coordinates = [(i, j, k) for (
                i, j, k), val in np.ndenumerate(cell.skeleton) if val != 0]
        else:
            skeleton_pixel_coordinates = [(i, j) for (
            i, j), val in np.ndenumerate(cell.skeleton) if val != 0]
        skeleton_pixel_coordinates = np.asarray(skeleton_pixel_coordinates)
        amin = np.argmin(np.linalg.norm(skeleton_pixel_coordinates-soma, axis=1))
        soma_on_skeleton = skeleton_pixel_coordinates[amin]

    return soma_on_skeleton


def get_total_length(skel):
    """Returns total length of skeleton in real world units.

    Parameters
    ----------
    skel : skan.csr.Skeleton
        A Skeleton object.
    """
    lengths = skel.path_lengths()
    return np.sum(lengths)


def get_avg_process_thickness(surface_area, total_length):
    return surface_area / total_length


def get_convex_hull(cell):
    convex_hull = convex_hull_image(cell.skeleton)
    cell.convex_hull = convex_hull

    return np.sum(convex_hull) * np.prod(cell.scale)


def get_no_of_forks(cell):
    # """Calculates # of forks by creating a binary structure
    # and counting all connected objects. Not the way it's done
    # in skan >=0.10"""
    # # get the degree for every cell pixel (no. of neighbouring pixels)
    # degrees = make_degree_image(cell.skeleton)
    # # array of all pixel locations with degree more than 2
    # fork_image = np.where(degrees > [2], 1, 0)
    # s = generate_binary_structure(cell.skeleton.ndim, 2)
    # num_forks = label(fork_image, structure=s)[1]

    # # for future plotting
    # fork_indices = np.where(degrees > [2])
    # if cell.skeleton.ndim == 2:
    #     cell._fork_coords = zip(fork_indices[0], fork_indices[1])
    # else:
    #     cell._fork_coords = zip(fork_indices[0], fork_indices[1],
    #                             fork_indices[2])
    skel = cell._skeleton
    mask = skel.degrees > 2
    num_forks = np.sum(mask)
    cell._fork_coords = skel.coordinates[mask]

    return num_forks


def _cmp(branch, path):
    edge = (path[0], path[-1])
    return (
        np.all(branch == edge)
        or np.all(branch == edge[::-1])
    )


def _branch_structure(junctions, branch_stats, paths_list):
    """
    junctions
    branches : array of float, shape (N, {4, 5})
        An array containing branch endpoint IDs, length, and branch type.
        The types are:
        - tip-tip (0)
        - tip-junction (1)
        - junction-junction (2)
        - path-path (3) (This can only be a standalone cycle)
    paths_list
    """
    next_set_junctions = []
    next_set_branches = []
    junc_to_junc = 2
    terminal = 1

    for junction in junctions:
        branches_travelled = []
        for branch_no, row in enumerate(branch_stats):
            src, dst, br_type = row
            if src == junction:  # check if start node ID of current branch equals junction
                if br_type == junc_to_junc:
                    next_set_junctions.append(dst)  # next process with junction as it's ending node
                    for path in paths_list:
                        if _cmp((src, dst), path):
                            next_set_branches.append(path)
                            branches_travelled.append(branch_no)
                if br_type == terminal:
                    for path in paths_list:
                        if _cmp((src, dst), path):
                            next_set_branches.append(path)
                            branches_travelled.append(branch_no)
            elif dst == junction:  # check if end node ID of current branch equals junction
                if br_type == junc_to_junc:
                    next_set_junctions.append(src)
                    for path in paths_list:
                        if _cmp((src, dst), path):
                            next_set_branches.append(path)
                            branches_travelled.append(branch_no)
                if br_type == terminal:
                    for path in paths_list:
                        if _cmp((src, dst), path):
                            next_set_branches.append(path)
                            branches_travelled.append(branch_no)

        branch_stats = np.delete(branch_stats, branches_travelled, axis=0)

    return next_set_junctions, next_set_branches, branch_stats


def _get_soma_node(skel, soma_on_skel):
    distances = np.linalg.norm(skel.coordinates - soma_on_skel, axis=1)
    soma_node = np.argmin(distances)
    return soma_node


def _get_soma_branches(soma_node, paths_list):
    soma_branches = []
    for path in paths_list:
        if soma_node in path: soma_branches.append(path)
    return soma_branches


def classify_branching_structure(skel, soma_on_skeleton):
    summary = summarize(skel)
    branch_stats = summary[['node-id-src', 'node-id-dst', 'branch-type']].to_numpy()
    paths_list = skel.paths_list()

    # get terminal branches
    term_branch_type = 1
    term_mask = summary['branch-type'] == term_branch_type
    term_branches = list(compress(skel.paths_list(), term_mask))

    branching_structure_array = []

    # get branches containing soma node
    soma_node = _get_soma_node(skel, soma_on_skeleton)
    soma_branches = _get_soma_branches([soma_node], paths_list)
    if len(soma_branches) > 2:
        # ID of the soma node (not lying on any branch: for skan < 0.11)
        # (lying on all fork branches: for skan >= 0.11)
        junctions = [soma_node]
        delete_soma_branch = False
    else:
        # collect first level/primary branches
        junctions = [soma_branches[0][0], soma_branches[0][-1]]  # ID (indices) of start & end node of soma_branch
        delete_soma_branch = True

    # eliminate loops in branches and path lists
    paths_list = _eliminate_loops(summary, paths_list)

    while True:
        junctions, branches, branch_stats = _branch_structure(
            junctions, branch_stats, paths_list)
        branching_structure_array.append(branches)
        if len(junctions) == 0:
            break

    if delete_soma_branch:
        branching_structure_array[0].remove(soma_branches[0])

    return branching_structure_array, term_branches


def get_primary_branches(branching_struct):
    prim_branches = branching_struct[0]
    n_prim_branches = len(prim_branches)
    avg_len_of_prim_branches = 0 if n_prim_branches == 0 else sum(
        map(len, prim_branches))/float(len(prim_branches))

    if n_prim_branches < 1:
        raise ValueError("Unable to detect any primary branches")

    return n_prim_branches, avg_len_of_prim_branches


def get_secondary_branches(branching_struct):
    sec_branches = [] if len(branching_struct) < 2 else branching_struct[1]

    n_sec_branches = len(sec_branches)
    avg_len_of_sec_branches = 0 if n_sec_branches == 0 else sum(
        map(len, sec_branches))/float(len(sec_branches))

    return n_sec_branches, avg_len_of_sec_branches


def get_tertiary_branches(branching_struct):
    tert_branches = [] if len(branching_struct) < 3 else branching_struct[2]

    n_tert_branches = len(tert_branches)
    avg_len_of_tert_branches = 0 if n_tert_branches == 0 else sum(
        map(len, tert_branches))/float(len(tert_branches))

    return n_tert_branches, avg_len_of_tert_branches


def get_quatenary_branches(branching_struct):
    quat_branches = [] if len(branching_struct) < 4 else branching_struct[3:]

    quat_branches = [
        branch for branch_lvl in quat_branches for branch in branch_lvl]
    n_quat_branches = len(quat_branches)
    avg_len_of_quat_branches = 0 if n_quat_branches == 0 else sum(
        map(len, quat_branches))/float(len(quat_branches))

    return n_quat_branches, avg_len_of_quat_branches


def get_terminal_branches(term_branches):
    n_term_branches = len(term_branches)
    avg_len_of_term_branches = 0 if n_term_branches == 0 else sum(
        map(len, term_branches))/float(len(term_branches))

    return n_term_branches, avg_len_of_term_branches
