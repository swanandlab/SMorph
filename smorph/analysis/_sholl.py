import numpy as np
import skan
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.spatial import distance_matrix
from skimage.draw import (
    ellipsoid,
)
from skimage.measure import (
    label,
    marching_cubes,
)
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import skew

from ..util._image import (
    _distance,
)


def sholl_analysis(
    shell_step_size,
    polynomial_degree,
    n_primary_branches,
    cell
):
    padded_skeleton = cell._padded_skeleton
    pad_sk_soma = cell._pad_sk_soma
    concentric_coords = None
    radii, n_intersections = get_intersections(
        shell_step_size, padded_skeleton, pad_sk_soma, cell)[-2:]

    (
        polynomial_model,
        polynomial_sholl_radii,
        predicted_n_intersections,
        non_zero_radii,
        non_zero_n_intersections
    ) = polynomial_fit(polynomial_degree, radii, n_intersections)

    (
        semi_log_r2,
        semi_log_regression_intercept,
        semi_log_regression_coeff
    ) = _semi_log(non_zero_n_intersections, non_zero_radii)

    (
        log_log_r2,
        log_log_regression_intercept,
        log_log_regression_coeff
    ) = _log_log(non_zero_n_intersections, non_zero_radii)

    determination_ratio = _get_determination_ratio(semi_log_r2, log_log_r2)

    norm_mthd = "Semi-log" if determination_ratio > 1 else "Log-log"

    critical_radius = get_critical_radius(
        non_zero_radii, predicted_n_intersections)

    critical_value = get_critical_value(predicted_n_intersections)

    enclosing_radius = get_enclosing_radius(
        non_zero_radii, non_zero_n_intersections)

    schoenen_ramification_index = get_schoenen_ramification_index(
        critical_value, n_primary_branches)

    skewness = get_skewness(polynomial_model, polynomial_degree,
                            non_zero_n_intersections)

    coefficient_of_determination = get_coefficient_of_determination(
        norm_mthd,
        semi_log_r2,
        log_log_r2
    )
    sholl_regression_coeff = get_sholl_regression_coeff(
        norm_mthd,
        semi_log_regression_coeff,
        log_log_regression_coeff
    )
    regression_intercept = get_regression_intercept(
        norm_mthd,
        semi_log_regression_intercept,
        log_log_regression_intercept
    )

    return (
        critical_radius,
        critical_value,
        enclosing_radius,
        schoenen_ramification_index,
        skewness,
        coefficient_of_determination,
        sholl_regression_coeff,
        regression_intercept,
        concentric_coords,
        n_intersections,
        polynomial_model,
        non_zero_n_intersections,
        polynomial_sholl_radii
    )


def _concentric_coords_and_values(
    shell_step_size,
    padded_skeleton,
    pad_sk_soma,
    largest_radius
):
    # concentric_coordinates: {radius values: [coords on that radius]}
    # n_intersections: {radius values: n_intersection values}

    # {100: [(10,10), ..] , 400: [(20,20), ..]}
    concentric_coords = defaultdict(list)
    concentric_coords_intensities = defaultdict(list)
    concentric_radii = [
        radius for radius in range(shell_step_size,
                                   largest_radius, shell_step_size)]

    if padded_skeleton.ndim == 2:
        for pt, value in np.ndenumerate(padded_skeleton):
            for radius in concentric_radii:
                lhs = _distance(pt, pad_sk_soma)
                if abs(lhs - radius) < 0.9:
                    concentric_coords[radius].append(pt)
                    concentric_coords_intensities[radius].append(value)
    else:
        for i, radius in enumerate(concentric_radii):
            el = ellipsoid(radius, radius, radius)
            center = tuple(map(lambda d: d//2, el.shape))

            vertse, facese, normalse, valuese = marching_cubes(el)

            sholl_sphere = vertse + i * shell_step_size + pad_sk_soma - (
                np.array(center) + i * shell_step_size)

            ball = np.zeros_like(padded_skeleton, dtype=int)
            bshp = ball.shape
            for j in sholl_sphere:
                pt = tuple(map(int, j))

                if (
                    pt[0] < bshp[0] and pt[1] < bshp[1] and pt[2] < bshp[2]
                    and pt[0] > 0 and pt[1] > 0 and pt[2] > 0
                ):
                    ball[pt] = 1

            intersections_mask = ball * padded_skeleton
            intersections_coords = np.argwhere(intersections_mask)
            for coord in intersections_coords:
                pt = tuple(coord)
                concentric_coords[radius].append(pt)
                concentric_coords_intensities[radius].append(len(intersections_coords))

    # array with intersection values corresponding to radii
    n_intersections = defaultdict()
    for radius, val in concentric_coords_intensities.items():
        intersec_indicies = []
        indexes = [i for i, x in enumerate(val) if x]
        for index in indexes:
            intersec_indicies.append(concentric_coords[radius][index])
        img = np.zeros(padded_skeleton.shape)

        for index in intersec_indicies:
            img[index] = 1
        label_image = label(img)
        n_intersections[radius] = np.amax(label_image)

    return concentric_coords, n_intersections


def _path_distances(skeleton, center_point, path_id):
    """Compute real world distances of specific skeleton path coordinates
    from the center point.

    Parameters
    ----------
    skeleton : skan.csr.Skeleton
        A Skeleton object.
    center_point : array
        Real world coordinates of center.
    path_id : int
        Path ID of path to be traversed.

    Returns
    -------
    ndarray
        Distance from each pixel in the path to the central pixel in real
        world units.
    """
    path = skeleton.path_coordinates(path_id)
    path_scaled = path * skeleton.spacing
    distances = np.ravel(distance_matrix(path_scaled, [center_point]))
    return distances


def get_intersections(
    shell_step_size,
    padded_skeleton,
    pad_sk_soma,
    cell,
    old=False
):
    """Sholl Analysis for Skeleton object.

    Parameters
    ----------
    skeleton : skan.csr.Skeleton
        A Skeleton object.
    center : array-like of float or None, optional
        Pixel coordinates of a point on the skeleton to use as the center
        from which the concentric shells are computed. If None, the
        geodesic center of skeleton is chosen.
    shells : int or array of floats or None, optional
        If an int, it is used as number of evenly spaced concentric shells. If
        an array of floats, it is used directly as the different shell radii in
        real world units. If None, the number of evenly spaced concentric
        shells is automatically calculated.

    Returns
    -------
    array
        Radii in real world units for concentric shells used for analysis.
    array
        Number of intersections for corresponding shell radii.
    """
    scale = np.asarray(cell.scale)
    skeleton = cell._skeleton
    center = np.asarray(cell.skel_soma)

    scaled_center = center * scale

    leaf_node_val = 1
    leaf_nodes_mask = np.argwhere(skeleton.degrees == leaf_node_val)
    leaf_nodes = np.squeeze(skeleton.coordinates[leaf_nodes_mask])
    leaf_to_center_vec = leaf_nodes - center
    leaf_to_center_px = np.linalg.norm(leaf_to_center_vec, axis=1)
    leaf_to_center_real = np.linalg.norm(
            leaf_to_center_vec * scale, axis=1
            )

    end_radius = np.max(leaf_to_center_real)  # largest possible radius

    if old:
        center = np.asarray(pad_sk_soma)
        # return sholl radii and corresponding intersection values
        xs, ys = [], []
        concentric_coords, n_intersections = _concentric_coords_and_values(
            shell_step_size, padded_skeleton,
            pad_sk_soma, end_radius)
        for rad, val in n_intersections.items():
            xs.append(rad)
            ys.append(val)
        order = np.argsort(xs)

        return (
            np.asarray(xs)[order],
            np.asarray(ys)[order]
        )

    shell_radii = np.arange(
        shell_step_size, end_radius+shell_step_size, shell_step_size)

    # width = np.linalg.norm(scale) / 2
    # shell_bins = [(r-width, r+width) for r in shell_radii]

    intersection_counts = np.zeros_like(shell_radii)

    for i in range(skeleton.n_paths):
        # Find distances of the path pixels
        distances = _path_distances(skeleton, scaled_center, i)

        # # Find which shell bin each pixel sits in
        # for j in range(len(shell_bins)):
        #     mn, mx = shell_bins[j]
        #     for distance in distances:
        #         if mn <= distance < mx:
        #             intersection_counts[j] += 1
        #             break

        # Find which shell bin each pixel sits in
        shell_location = np.digitize(distances, shell_radii)

        # Use np.diff to find where bins are crossed.
        crossings = shell_location[
            np.flatnonzero(np.diff(shell_location))]

        # increment corresponding crossings
        intersection_counts[crossings] += 1

    # print(shell_radii, intersection_counts)
    return shell_radii, intersection_counts


def polynomial_fit(polynomial_degree, radii, n_intersections):
    """Models relationship between intersections & radii."""

    # till last non-zero value
    last_intersection_index = np.max(np.nonzero(n_intersections))

    if last_intersection_index == 0:
        raise ValueError('Sholl analysis found no branch intersections!')

    non_zero_n_intersections = n_intersections[:last_intersection_index+1]
    non_zero_radii = radii[:last_intersection_index+1]

    y_data = non_zero_n_intersections
    reshaped_x = non_zero_radii.reshape((-1, 1))

    x_ = PolynomialFeatures(
        degree=polynomial_degree, include_bias=False).fit_transform(reshaped_x)
    # create a linear regression model
    polynomial_model = LinearRegression().fit(x_, y_data)

    predicted_n_intersections = polynomial_model.predict(x_)

    return (
        polynomial_model,
        x_,
        predicted_n_intersections,
        non_zero_radii,
        non_zero_n_intersections
    )


def get_enclosing_radius(non_zero_radii, non_zero_n_intersections):
    """Index of last non-zero value in the array containing radii."""
    n_intersected_circles = len(non_zero_n_intersections)
    enclosing_circle = (non_zero_n_intersections != 0)[::-1].argmax()
    return non_zero_radii[n_intersected_circles - enclosing_circle - 1]


def get_critical_radius(non_zero_radii, predicted_n_intersections):
    return non_zero_radii[np.argmax(predicted_n_intersections)]


def get_critical_value(predicted_n_intersections):
    # local maximum of the polynomial fit (Maximum no. of intersections)
    return np.max(predicted_n_intersections)


def get_skewness(
    polynomial_model,
    polynomial_degree,
    non_zero_n_intersections
):
    # Indication of how symmetrical polynomial distribution is around its mean.
    reshaped_x = non_zero_n_intersections.reshape((-1, 1))
    x_ = PolynomialFeatures(
        degree=polynomial_degree, include_bias=False).fit_transform(reshaped_x)
    return skew(polynomial_model.predict(x_))


def get_schoenen_ramification_index(critical_value, n_primary_branches):
    # ratio between critical value and number of primary branches
    schoenen_ramification_index = critical_value / n_primary_branches
    return schoenen_ramification_index


def _semi_log(non_zero_n_intersections, non_zero_radii):
    # no. of intersections/circumference
    normalized_y = np.log(
        non_zero_n_intersections/(2*np.pi*non_zero_radii))
    reshaped_x = non_zero_radii.reshape((-1, 1))
    model = LinearRegression().fit(reshaped_x, normalized_y)

    # predict y from the data
    model.predict(reshaped_x)
    r2 = model.score(reshaped_x, normalized_y)
    regression_intercept = model.intercept_
    regression_coefficient = -model.coef_[0]

    return r2, regression_intercept, regression_coefficient


def _log_log(non_zero_n_intersections, non_zero_radii):
    # no. of intersections/circumference
    normalized_y = np.log(
        non_zero_n_intersections/(2*np.pi*non_zero_radii))
    reshaped_x = non_zero_radii.reshape((-1, 1))
    normalized_x = np.log(reshaped_x)
    model = LinearRegression().fit(normalized_x, normalized_y)

    # predict y from the data
    model.predict(normalized_x)
    r2 = model.score(normalized_x, normalized_y)
    regression_intercept = model.intercept_
    regression_coefficient = -model.coef_[0]

    return r2, regression_intercept, regression_coefficient


def _get_determination_ratio(semi_log_r2, log_log_r2):
    return semi_log_r2 / log_log_r2


def get_coefficient_of_determination(norm_mthd, semi_log_r2, log_log_r2):
    # how close the data are to the fitted regression (indicative of the level
    # of explained variability in the data set)
    if norm_mthd == "Semi-log":
        return semi_log_r2
    return log_log_r2


def get_regression_intercept(
    norm_mthd,
    semi_log_regression_intercept,
    log_log_regression_intercept
):
    # Y intercept of the logarithmic plot
    if norm_mthd == "Semi-log":
        return semi_log_regression_intercept
    return log_log_regression_intercept


def get_sholl_regression_coeff(
    norm_mthd,
    semi_log_regression_coeff,
    log_log_regression_coeff
):
    """Rate of decay of no. of branches."""
    if norm_mthd == "Semi-log":
        return semi_log_regression_coeff
    return log_log_regression_coeff
