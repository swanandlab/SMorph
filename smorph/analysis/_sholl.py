import numpy as np
from collections import defaultdict
from skimage.measure import label
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import skew


def sholl_analysis(
    shell_step_size,
    polynomial_degree,
    largest_radius,
    padded_skeleton,
    pad_sk_soma,
    n_primary_branches
):
    concentric_coords, radii, n_intersections = get_intersections(
        shell_step_size, padded_skeleton, pad_sk_soma, largest_radius)

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
        radii,
        n_intersections,
        polynomial_model,
        non_zero_radii,
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
    concentric_coordinates = defaultdict(list)
    concentric_coordinates_intensities = defaultdict(list)
    concentric_radiuses = [
        radius for radius in range(shell_step_size,
                                   largest_radius, shell_step_size)]

    for (x, y), value in np.ndenumerate(padded_skeleton):
        for radius in concentric_radiuses:
            lhs = (x - pad_sk_soma[0])**2 + (y - pad_sk_soma[1])**2
            if abs((np.sqrt(lhs)-radius)) < 0.9:
                concentric_coordinates[radius].append((x, y))
                concentric_coordinates_intensities[radius].append(value)

    # array with intersection values corresponding to radii
    n_intersections = defaultdict()
    for radius, val in concentric_coordinates_intensities.items():
        intersec_indicies = []
        indexes = [i for i, x in enumerate(val) if x]
        for index in indexes:
            intersec_indicies.append(concentric_coordinates[radius][index])
        img = np.zeros(padded_skeleton.shape)

        for index in intersec_indicies:
            img[index] = 1
        label_image = label(img)
        n_intersections[radius] = np.amax(label_image)

    return concentric_coordinates, n_intersections


def get_intersections(
    shell_step_size,
    padded_skeleton,
    pad_sk_soma,
    largest_radius
):
    # return sholl radii and corresponding intersection values
    xs, ys = [], []
    concentric_coords, n_intersections = _concentric_coords_and_values(
        shell_step_size, padded_skeleton,
        pad_sk_soma, largest_radius)
    for rad, val in n_intersections.items():
        xs.append(rad)
        ys.append(val)
    order = np.argsort(xs)

    return concentric_coords, np.array(xs)[order], np.array(ys)[order]


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
    return round(np.max(predicted_n_intersections), 2)


def get_skewness(
    polynomial_model,
    polynomial_degree,
    non_zero_n_intersections
):
    # Indication of how symmetrical polynomial distribution is around its mean.
    reshaped_x = non_zero_n_intersections.reshape((-1, 1))
    x_ = PolynomialFeatures(
        degree=polynomial_degree, include_bias=False).fit_transform(reshaped_x)
    return round(skew(polynomial_model.predict(x_)), 2)


def get_schoenen_ramification_index(critical_value, n_primary_branches):
    # ratio between critical value and number of primary branches
    schoenen_ramification_index = critical_value / n_primary_branches
    return round(schoenen_ramification_index, 2)


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
        return round(semi_log_r2, 2)
    else:
        return round(log_log_r2, 2)


def get_regression_intercept(
    norm_mthd,
    semi_log_regression_intercept,
    log_log_regression_intercept
):
    # Y intercept of the logarithmic plot
    if norm_mthd == "Semi-log":
        return round(semi_log_regression_intercept, 2)
    else:
        return round(log_log_regression_intercept, 2)


def get_sholl_regression_coeff(
    norm_mthd,
    semi_log_regression_coeff,
    log_log_regression_coeff
):
    """Rate of decay of no. of branches."""
    if norm_mthd == "Semi-log":
        return round(semi_log_regression_coeff, 2)
    else:
        return round(log_log_regression_coeff, 2)
