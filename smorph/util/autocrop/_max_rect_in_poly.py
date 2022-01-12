import numpy as np
import cvxpy
from shapely.geometry import Polygon


def _get_intersection(coords):
    """Given an input list of coordinates, find the intersection
    section of corner coordinates. Returns geojson of the
    interesection polygon.
    """
    ipoly = None
    for coord in coords:
        if ipoly is None:
            ipoly = Polygon(coord)
        else:
            tmp = Polygon(coord)
            ipoly = ipoly.intersection(tmp)

    # close the polygon loop by adding the first coordinate again
    first_x = ipoly.exterior.coords.xy[0][0]
    first_y = ipoly.exterior.coords.xy[1][0]
    ipoly.exterior.coords.xy[0].append(first_x)
    ipoly.exterior.coords.xy[1].append(first_y)

    inter_coords = zip(
        ipoly.exterior.coords.xy[0], ipoly.exterior.coords.xy[1])

    return inter_coords


def __two_pts_to_line(pt1, pt2):
    """Create a line from two points in form of
    a1(x) + a2(y) = b
    """
    pt1 = [float(p) for p in pt1]
    pt2 = [float(p) for p in pt2]
    try:
        slp = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
    except ZeroDivisionError:
        slp = 1e5 * (pt2[1] - pt1[1])
    a1 = -slp
    a2 = 1.
    b = -slp * pt1[0] + pt1[1]

    return a1, a2, b


def _pts_to_leq(coords):
    """Converts a set of points to form Ax = b, but since
    x is of length 2 this is like A1(x1) + A2(x2) = B.
    returns A1, A2, B
    """
    A1 = []
    A2 = []
    B = []
    for i in range(len(coords) - 1):
        pt1 = coords[i]
        pt2 = coords[i + 1]
        a1, a2, b = __two_pts_to_line(pt1, pt2)
        A1.append(a1)
        A2.append(a2)
        B.append(b)
    return A1, A2, B


def get_maximal_rectangle(coords, point=None):
    """Find the largest, inscribed, axis-aligned rectangle.
    :param coordinates:
        A list of of [x, y] pairs describing a closed, convex polygon.
    """
    coordinates = _get_intersection(coords)
    coordinates = np.array((list(coordinates)))

    x_range = np.max(coordinates, axis=0)[0]-np.min(coordinates, axis=0)[0]
    y_range = np.max(coordinates, axis=0)[1]-np.min(coordinates, axis=0)[1]

    scale = np.array([x_range, y_range])
    sc_coordinates = coordinates/scale

    poly = Polygon(sc_coordinates)

    inside_pt = ((poly.representative_point().x,
                 poly.representative_point().y) if point is None else sc_coordinates[point])

    A1, A2, B = _pts_to_leq(sc_coordinates)

    bl = cvxpy.Variable(2)  # bottom left
    tr = cvxpy.Variable(2)  # top right
    br = cvxpy.Variable(2)  # bottom right
    tl = cvxpy.Variable(2)  # top left
    obj = cvxpy.Maximize(cvxpy.log(tr[0] - bl[0]) + cvxpy.log(bl[1] - tr[1]))  # (tr[0] - bl[0]) * (bl[1] - tr[1]))
    constraints = [bl[0] == tl[0], br[0] == tr[0],
                   tl[1] == tr[1], bl[1] == br[1]]

    for i in range(len(B)):
        if inside_pt[0] * A1[i] + inside_pt[1] * A2[i] <= B[i]:
            constraints.append(bl[0] * A1[i] + bl[1] * A2[i] <= B[i])
            constraints.append(tr[0] * A1[i] + tr[1] * A2[i] <= B[i])
            constraints.append(br[0] * A1[i] + br[1] * A2[i] <= B[i])
            constraints.append(tl[0] * A1[i] + tl[1] * A2[i] <= B[i])
        else:
            constraints.append(bl[0] * A1[i] + bl[1] * A2[i] >= B[i])
            constraints.append(tr[0] * A1[i] + tr[1] * A2[i] >= B[i])
            constraints.append(br[0] * A1[i] + br[1] * A2[i] >= B[i])
            constraints.append(tl[0] * A1[i] + tl[1] * A2[i] >= B[i])

    prob = cvxpy.Problem(obj, constraints)
    # ECOS, CVXOPT, SCS, SCIPY
    prob.solve()

    # print(bl.value, tr.value, br.value, tl.value)
    bottom_left = np.array(bl.value).T * scale
    top_right = np.array(tr.value).T * scale
    # print(bottom_left, top_right)

    return bottom_left, top_right