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


# def get_maximal_rectangle(coords, point=None):
#     """Find the largest, inscribed, axis-aligned rectangle.
#     :param coordinates:
#         A list of of [x, y] pairs describing a closed, convex polygon.
#     """
#     coordinates = _get_intersection(coords)
#     coordinates = np.array((list(coordinates)))

#     x_range = np.max(coordinates, axis=0)[0]-np.min(coordinates, axis=0)[0]
#     y_range = np.max(coordinates, axis=0)[1]-np.min(coordinates, axis=0)[1]

#     scale = np.array([x_range, y_range])
#     sc_coordinates = coordinates/scale

#     poly = Polygon(sc_coordinates)

#     inside_pt = ((poly.representative_point().x,
#                  poly.representative_point().y) if point is None else sc_coordinates[point])

#     A1, A2, B = _pts_to_leq(sc_coordinates)

#     bl = cvxpy.Variable(2)  # bottom left
#     tr = cvxpy.Variable(2)  # top right
#     br = cvxpy.Variable(2)  # bottom right
#     tl = cvxpy.Variable(2)  # top left
#     obj = cvxpy.Maximize(cvxpy.log(tr[0] - bl[0]) + cvxpy.log(bl[1] - tr[1]))  # (tr[0] - bl[0]) * (bl[1] - tr[1]))
#     constraints = [bl[0] == tl[0], br[0] == tr[0],
#                    tl[1] == tr[1], bl[1] == br[1]]

#     for i in range(len(B)):
#         if inside_pt[0] * A1[i] + inside_pt[1] * A2[i] <= B[i]:
#             constraints.append(bl[0] * A1[i] + bl[1] * A2[i] <= B[i])
#             constraints.append(tr[0] * A1[i] + tr[1] * A2[i] <= B[i])
#             constraints.append(br[0] * A1[i] + br[1] * A2[i] <= B[i])
#             constraints.append(tl[0] * A1[i] + tl[1] * A2[i] <= B[i])
#         else:
#             constraints.append(bl[0] * A1[i] + bl[1] * A2[i] >= B[i])
#             constraints.append(tr[0] * A1[i] + tr[1] * A2[i] >= B[i])
#             constraints.append(br[0] * A1[i] + br[1] * A2[i] >= B[i])
#             constraints.append(tl[0] * A1[i] + tl[1] * A2[i] >= B[i])

#     prob = cvxpy.Problem(obj, constraints)
#     # ECOS, CVXOPT, SCS, SCIPY
#     prob.solve()

#     # print(bl.value, tr.value, br.value, tl.value)
#     bottom_left = np.array(bl.value).T * scale
#     top_right = np.array(tr.value).T * scale
#     # print(bottom_left, top_right)

#     return bottom_left, top_right


def get_maximal_rectangle(mask):
    def make_h(binary_image2d):
        result = np.zeros_like(binary_image2d, dtype=int)
        result[0] = binary_image2d[0]
        for i in range(1, result.shape[0]):
            result[i] = np.where(binary_image2d[i], result[i-1] + 1, 0)
        return result


    class Node:
        def __init__(self, value, index):
            self.value = value
            self.index = index
            self.left = None
            self.right = None

        def __repr__(self):
            return f'Node(value={self.value}, index={self.index})'

        def __str__(self):
            return self.__repr__()


    def make_tree(arr):
        stack = []
        nodes = []
        for i, elem in enumerate(arr):
            new_node = Node(elem, i)
            last = None
            while stack:
                top = stack[-1]
                if top.value > elem:
                    last = stack.pop()
                else:
                    break
            if last is not None:
                new_node.left = last
            if stack:
                stack[-1].right = new_node
            stack.append(new_node)
        return stack[0]


    # note: we could bias this towards more square rectangles if we
    # use some measure of "squareness" for the second value in the tuple
    def largest_rectangle(low, high, root):
        if low == high or root is None:
            return (0,)
        return max(
            (root.value * (high - low), root.value, low, high),
            largest_rectangle(low, root.index, root.left),
            largest_rectangle(root.index + 1, high, root.right),
            )


    # find the largest rectangle in a 2D binary mask, return its coordinates
    # as slices
    # uses https://stackoverflow.com/a/12387148/224254
    # and https://stackoverflow.com/a/50651622/224254
    def largest_rectangle_in_mask(binary_mask2d):
        h = make_h(binary_mask2d)
        max_rect_size = 0
        for i, row in enumerate(h):
            tree = make_tree(row)
            max_rect_ending_in_row = largest_rectangle(
                0, row.size, tree
                )
            size, height, curr_start_col, curr_end_col = (
                max_rect_ending_in_row
                )
            if size > max_rect_size:
                start_row = i - height + 1
                end_row = i + 1
                start_col = curr_start_col
                end_col = curr_end_col
                max_rect_size = size
        return (start_row, end_row, start_col, end_col)


    return largest_rectangle_in_mask(mask)
