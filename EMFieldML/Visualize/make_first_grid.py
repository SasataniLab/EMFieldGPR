"""First-level grid generation for electromagnetic field visualization.

This module provides functionality to create the initial grid structure
for electromagnetic field analysis and visualization.


"""

import sys
from pathlib import Path

import numpy as np

from EMFieldML.config import config
from EMFieldML.Modeling import default_config
from EMFieldML.Modeling.make_first_grid import Point
from EMFieldML.Visualize import Deformation


class Point_outside:
    """Information for outside prediction points.

    This class represents prediction points outside the ferrite shield.

    Attributes:
        xyz (ndarray(3)): Point coordinates
        index (int): Index of the corresponding point on the ferrite shield surface
        mode (int): Point type: 0=prediction, 1=TX, 2=RX
        default_point (ndarray(3)): Original coordinates before movement (used for weight calculation when ferrite shape changes)
        weight (ndarray(n)): Weights for linear blend skinning with ferrite shield points
        parent (ndarray(3)): Index of connected points (original points)
        edges (list): Edge indices extending from this point (for easier search)
        planes (list): Plane indices this point belongs to (for easier search)

    """

    def __init__(self, point):
        """Initialize point with coordinates."""
        self.xyz = np.zeros(3)
        self.index = -1
        self.mode = 0  # 0: 外部、1:TX、2:RX
        for i in range(3):
            self.xyz[i] = point[i]
        self.default_point = np.copy(point)
        self.weight = self.calculate_weight(point)
        self.parent = []
        self.edges = []
        self.planes = []
        self.affilication = -1

    def calculate_weight(self, point):
        """Calculate weight for linear blend skinning."""
        points_default = default_config.points_default

        weight_sum = 0
        min_limit = 1e-8

        all_d = []
        for i in range(config.n_initial_points // 2):
            d = np.linalg.norm(points_default[i] - point, ord=2)
            all_d.append(d)
            if d < min_limit:
                weight_sum = np.inf
            else:
                weight_sum += 1 / d ** (config.LBS_w_make)
        for i in range(config.n_initial_points // 2):
            d = np.linalg.norm(
                np.array([points_default[i][0], points_default[i][1], -0.06]) - point,
                ord=2,
            )
            all_d.append(d)
            if d < min_limit:
                weight_sum = np.inf
            else:
                weight_sum += 1 / d ** (config.LBS_w_make)
        for i in range(config.n_initial_points // 2):
            d = np.linalg.norm(
                np.array([points_default[i][0], points_default[i][1], 0.05]) - point,
                ord=2,
            )
            all_d.append(d)
            if d < min_limit:
                weight_sum = np.inf
            else:
                weight_sum += 1 / d ** (config.LBS_w_make)
        for i in range(config.n_initial_points // 2):
            d = np.linalg.norm(
                np.array([points_default[i][0], points_default[i][1], 0.11]) - point,
                ord=2,
            )
            all_d.append(d)
            if d < min_limit:
                weight_sum = np.inf
            else:
                weight_sum += 1 / d ** (config.LBS_w_make)

        for i in range(config.n_initial_points * 2):
            if all_d[i] < min_limit:
                all_d[i] = 1
            else:
                all_d[i] = (1 / all_d[i] ** (config.LBS_w_make)) / weight_sum

        return np.array(all_d)

    def calculate_nextpoint(self, changes):
        """Calculate next point position with changes."""
        self.xyz = np.copy(self.default_point)
        self.xyz += np.sum(changes * self.weight.reshape(-1, 1), axis=0)


class Edge:
    """Edge class for visualization."""

    def __init__(self, point1_index, point2_index):
        """Initialize edge between two points."""
        self.node = [point1_index, point2_index]
        self.children = [-1, -1]


class Edge_outside:
    """Edge class for outside points in visualization."""

    def __init__(self, index1, index2):
        """Initialize outside edge between two points."""
        self.node = [index1, index2]
        self.children = [-1, -1]
        self.center = -1


class Plane:
    """Plane class for visualization."""

    def __init__(self, edges, points_index):
        """Initialize plane with four points."""
        self.node = [points_index[0], points_index[1], points_index[2], points_index[3]]
        self.edge = [-1, -1, -1, -1]
        for i in range(4):
            j = (i + 1) % 4
            for k in range(len(edges)):
                if (
                    edges[k].node[0] == points_index[i]
                    and edges[k].node[1] == points_index[j]
                ):
                    self.edge[i] = k
                if (
                    edges[k].node[0] == points_index[j]
                    and edges[k].node[1] == points_index[i]
                ):
                    self.edge[i] = k
            if self.edge[i] == -1:
                print(points_index)
                sys.exit(["make invalid plane."])
        self.children = [-1, -1, -1, -1]


class Plane_outside:
    """Plane class for outside points in visualization."""

    def __init__(self, index1, index2, index3, index4):
        """Initialize outside plane with four points."""
        self.node = [index1, index2, index3, index4]
        self.edge = [-1, -1, -1, -1]
        self.children = [-1, -1, -1, -1]
        self.center = -1


class Box:
    """Box class for visualization."""

    def __init__(self, points_index):
        """Initialize box with eight corner points."""
        self.node = [
            points_index[0],
            points_index[1],
            points_index[2],
            points_index[3],
            points_index[4],
            points_index[5],
            points_index[6],
            points_index[7],
        ]
        self.children = [-1 for _ in range(8)]
        self.edge_node = []
        self.plane_node = []


def initial_point(i, data_list):
    """Create initial point from data list.

    Args:
        i: Point index
        data_list: List containing coordinate data

    Returns:
        Point: Initialized point object

    """
    point = Point(
        np.array([data_list[3 * i + 0], data_list[3 * i + 1], data_list[3 * i + 2]])
    )
    point.corner = 1
    return point


def initial_plane_point(i, data_list):
    """Create initial plane point from data list.

    Args:
        i: Point index
        data_list: List containing coordinate data

    Returns:
        Point: Initialized point object

    """
    return Point(
        np.array([data_list[3 * i + 0], data_list[3 * i + 1], data_list[3 * i + 2]])
    )


def make_first_point_edge_plane(
    input_path: Path,
):
    """Create first-level grid points, edges, and planes."""
    points = [None for _ in range(72)]
    edges = [None for _ in range(140)]
    planes = [None for _ in range(70)]

    with Path(input_path).open() as f:
        data = f.readlines()
        coordinate_data = [float(row.rstrip("\n")) for row in data]  #

    for i in range(20):
        points[i] = initial_point(i, coordinate_data)
        points[i + 36] = initial_point(i + 36, coordinate_data)
    for i in range(20, 36):
        points[i] = initial_plane_point(i, coordinate_data)
        points[i + 36] = initial_plane_point(i + 36, coordinate_data)

    edges[0] = Edge(0, 1)
    edges[1] = Edge(1, 2)
    edges[2] = Edge(2, 3)
    edges[3] = Edge(3, 4)
    edges[4] = Edge(4, 5)
    edges[5] = Edge(19, 20)
    edges[6] = Edge(20, 21)
    edges[7] = Edge(21, 22)
    edges[8] = Edge(22, 23)
    edges[9] = Edge(23, 6)
    edges[10] = Edge(18, 31)
    edges[11] = Edge(31, 32)
    edges[12] = Edge(32, 33)
    edges[13] = Edge(33, 24)
    edges[14] = Edge(24, 7)
    edges[15] = Edge(17, 30)
    edges[16] = Edge(30, 35)
    edges[17] = Edge(35, 34)
    edges[18] = Edge(34, 25)
    edges[19] = Edge(25, 8)
    edges[20] = Edge(16, 29)
    edges[21] = Edge(29, 28)
    edges[22] = Edge(28, 27)
    edges[23] = Edge(27, 26)
    edges[24] = Edge(26, 9)
    edges[25] = Edge(15, 14)
    edges[26] = Edge(14, 13)
    edges[27] = Edge(13, 12)
    edges[28] = Edge(12, 11)
    edges[29] = Edge(11, 10)
    edges[30] = Edge(0, 19)
    edges[31] = Edge(19, 18)
    edges[32] = Edge(18, 17)
    edges[33] = Edge(17, 16)
    edges[34] = Edge(16, 15)
    edges[35] = Edge(1, 20)
    edges[36] = Edge(20, 31)
    edges[37] = Edge(31, 30)
    edges[38] = Edge(30, 29)
    edges[39] = Edge(29, 14)
    edges[40] = Edge(2, 21)
    edges[41] = Edge(21, 32)
    edges[42] = Edge(32, 35)
    edges[43] = Edge(35, 28)
    edges[44] = Edge(28, 13)
    edges[45] = Edge(3, 22)
    edges[46] = Edge(22, 33)
    edges[47] = Edge(33, 34)
    edges[48] = Edge(34, 27)
    edges[49] = Edge(27, 12)
    edges[50] = Edge(4, 23)
    edges[51] = Edge(23, 24)
    edges[52] = Edge(24, 25)
    edges[53] = Edge(25, 26)
    edges[54] = Edge(26, 11)
    edges[55] = Edge(5, 6)
    edges[56] = Edge(6, 7)
    edges[57] = Edge(7, 8)
    edges[58] = Edge(8, 9)
    edges[59] = Edge(9, 10)
    edges[60] = Edge(0, 36)
    edges[61] = Edge(1, 37)
    edges[62] = Edge(2, 38)
    edges[63] = Edge(3, 39)
    edges[64] = Edge(4, 40)
    edges[65] = Edge(5, 41)
    edges[66] = Edge(6, 42)
    edges[67] = Edge(7, 43)
    edges[68] = Edge(8, 44)
    edges[69] = Edge(9, 45)
    edges[70] = Edge(10, 46)
    edges[71] = Edge(11, 47)
    edges[72] = Edge(12, 48)
    edges[73] = Edge(13, 49)
    edges[74] = Edge(14, 50)
    edges[75] = Edge(15, 51)
    edges[76] = Edge(16, 52)
    edges[77] = Edge(17, 53)
    edges[78] = Edge(18, 54)
    edges[79] = Edge(19, 55)
    edges[80] = Edge(36, 37)
    edges[81] = Edge(37, 38)
    edges[82] = Edge(38, 39)
    edges[83] = Edge(39, 40)
    edges[84] = Edge(40, 41)
    edges[85] = Edge(55, 56)
    edges[86] = Edge(56, 57)
    edges[87] = Edge(57, 58)
    edges[88] = Edge(58, 59)
    edges[89] = Edge(59, 42)
    edges[90] = Edge(54, 67)
    edges[91] = Edge(67, 68)
    edges[92] = Edge(68, 69)
    edges[93] = Edge(69, 60)
    edges[94] = Edge(60, 43)
    edges[95] = Edge(53, 66)
    edges[96] = Edge(66, 71)
    edges[97] = Edge(71, 70)
    edges[98] = Edge(70, 61)
    edges[99] = Edge(61, 44)
    edges[100] = Edge(52, 65)
    edges[101] = Edge(65, 64)
    edges[102] = Edge(64, 63)
    edges[103] = Edge(63, 62)
    edges[104] = Edge(62, 45)
    edges[105] = Edge(51, 50)
    edges[106] = Edge(50, 49)
    edges[107] = Edge(49, 48)
    edges[108] = Edge(48, 47)
    edges[109] = Edge(47, 46)
    edges[110] = Edge(36, 55)
    edges[111] = Edge(55, 54)
    edges[112] = Edge(54, 53)
    edges[113] = Edge(53, 52)
    edges[114] = Edge(52, 51)
    edges[115] = Edge(37, 56)
    edges[116] = Edge(56, 67)
    edges[117] = Edge(67, 66)
    edges[118] = Edge(66, 65)
    edges[119] = Edge(65, 50)
    edges[120] = Edge(38, 57)
    edges[121] = Edge(57, 68)
    edges[122] = Edge(68, 71)
    edges[123] = Edge(71, 64)
    edges[124] = Edge(64, 49)
    edges[125] = Edge(39, 58)
    edges[126] = Edge(58, 69)
    edges[127] = Edge(69, 70)
    edges[128] = Edge(70, 63)
    edges[129] = Edge(63, 48)
    edges[130] = Edge(40, 59)
    edges[131] = Edge(59, 60)
    edges[132] = Edge(60, 61)
    edges[133] = Edge(61, 62)
    edges[134] = Edge(62, 47)
    edges[135] = Edge(41, 42)
    edges[136] = Edge(42, 43)
    edges[137] = Edge(43, 44)
    edges[138] = Edge(44, 45)
    edges[139] = Edge(45, 46)

    planes[0] = Plane(edges, [0, 1, 20, 19])
    planes[1] = Plane(edges, [1, 2, 21, 20])
    planes[2] = Plane(edges, [2, 3, 22, 21])
    planes[3] = Plane(edges, [3, 4, 23, 22])
    planes[4] = Plane(edges, [4, 5, 6, 23])
    planes[5] = Plane(edges, [19, 20, 31, 18])
    planes[6] = Plane(edges, [20, 21, 32, 31])
    planes[7] = Plane(edges, [21, 22, 33, 32])
    planes[8] = Plane(edges, [22, 23, 24, 33])
    planes[9] = Plane(edges, [23, 6, 7, 24])
    planes[10] = Plane(edges, [18, 31, 30, 17])
    planes[11] = Plane(edges, [31, 32, 35, 30])
    planes[12] = Plane(edges, [32, 33, 34, 35])
    planes[13] = Plane(edges, [33, 24, 25, 34])
    planes[14] = Plane(edges, [24, 7, 8, 25])
    planes[15] = Plane(edges, [17, 30, 29, 16])
    planes[16] = Plane(edges, [30, 35, 28, 29])
    planes[17] = Plane(edges, [35, 34, 27, 28])
    planes[18] = Plane(edges, [34, 25, 26, 27])
    planes[19] = Plane(edges, [25, 8, 9, 26])
    planes[20] = Plane(edges, [16, 29, 14, 15])
    planes[21] = Plane(edges, [29, 28, 13, 14])
    planes[22] = Plane(edges, [28, 27, 12, 13])
    planes[23] = Plane(edges, [27, 26, 11, 12])
    planes[24] = Plane(edges, [26, 9, 10, 11])
    planes[25] = Plane(edges, [0, 36, 37, 1])
    planes[26] = Plane(edges, [1, 37, 38, 2])
    planes[27] = Plane(edges, [2, 38, 39, 3])
    planes[28] = Plane(edges, [3, 39, 40, 4])
    planes[29] = Plane(edges, [4, 40, 41, 5])
    planes[30] = Plane(edges, [5, 41, 42, 6])
    planes[31] = Plane(edges, [6, 42, 43, 7])
    planes[32] = Plane(edges, [7, 43, 44, 8])
    planes[33] = Plane(edges, [8, 44, 45, 9])
    planes[34] = Plane(edges, [9, 45, 46, 10])
    planes[35] = Plane(edges, [10, 46, 47, 11])
    planes[36] = Plane(edges, [11, 47, 48, 12])
    planes[37] = Plane(edges, [12, 48, 49, 13])
    planes[38] = Plane(edges, [13, 49, 50, 14])
    planes[39] = Plane(edges, [14, 50, 51, 15])
    planes[40] = Plane(edges, [15, 51, 52, 16])
    planes[41] = Plane(edges, [16, 52, 53, 17])
    planes[42] = Plane(edges, [17, 53, 54, 18])
    planes[43] = Plane(edges, [18, 54, 55, 19])
    planes[44] = Plane(edges, [19, 55, 36, 0])
    planes[45] = Plane(edges, [36, 55, 56, 37])
    planes[46] = Plane(edges, [37, 56, 57, 38])
    planes[47] = Plane(edges, [38, 57, 58, 39])
    planes[48] = Plane(edges, [39, 58, 59, 40])
    planes[49] = Plane(edges, [40, 59, 42, 41])
    planes[50] = Plane(edges, [55, 54, 67, 56])
    planes[51] = Plane(edges, [56, 67, 68, 57])
    planes[52] = Plane(edges, [57, 68, 69, 58])
    planes[53] = Plane(edges, [58, 69, 60, 59])
    planes[54] = Plane(edges, [59, 60, 43, 42])
    planes[55] = Plane(edges, [54, 53, 66, 67])
    planes[56] = Plane(edges, [67, 66, 71, 68])
    planes[57] = Plane(edges, [68, 71, 70, 69])
    planes[58] = Plane(edges, [69, 70, 61, 60])
    planes[59] = Plane(edges, [60, 61, 44, 43])
    planes[60] = Plane(edges, [53, 52, 65, 66])
    planes[61] = Plane(edges, [66, 65, 64, 71])
    planes[62] = Plane(edges, [71, 64, 63, 70])
    planes[63] = Plane(edges, [70, 63, 62, 61])
    planes[64] = Plane(edges, [61, 62, 45, 44])
    planes[65] = Plane(edges, [52, 51, 50, 65])
    planes[66] = Plane(edges, [65, 50, 49, 64])
    planes[67] = Plane(edges, [64, 49, 48, 63])
    planes[68] = Plane(edges, [63, 48, 47, 62])
    planes[69] = Plane(edges, [62, 47, 46, 45])

    return points, edges, planes


def make_first_out_points(points, planes):
    """Create first-level out points for grid generation."""
    x = np.array([-0.40, -0.325, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.325, 0.40])
    y = np.array([-0.40, -0.325, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.325, 0.40])
    z = np.array([-0.20, -0.13, -0.06, 0.0, 0.025, 0.05, 0.11, 0.205, 0.3])
    out_points = []
    for i in range(10):
        for j in range(10):
            for k in range(9):
                out_point = Point_outside(np.array([x[i], y[j], z[k]]))
                if i == 2 and j == 2 and k == 2:
                    out_point.mode = 1
                    out_point.index = 41
                elif i == 2 and j == 2 and k == 3:
                    out_point.mode = 1
                    out_point.index = 5
                elif i == 2 and j == 2 and k == 5:
                    out_point.mode = 2
                    out_point.index = 5
                elif i == 2 and j == 2 and k == 6:
                    out_point.mode = 2
                    out_point.index = 41
                elif i == 2 and j == 3 and k == 2:
                    out_point.mode = 1
                    out_point.index = 40
                elif i == 2 and j == 3 and k == 3:
                    out_point.mode = 1
                    out_point.index = 4
                elif i == 2 and j == 3 and k == 5:
                    out_point.mode = 2
                    out_point.index = 4
                elif i == 2 and j == 3 and k == 6:
                    out_point.mode = 2
                    out_point.index = 40
                elif i == 2 and j == 4 and k == 2:
                    out_point.mode = 1
                    out_point.index = 39
                elif i == 2 and j == 4 and k == 3:
                    out_point.mode = 1
                    out_point.index = 3
                elif i == 2 and j == 4 and k == 5:
                    out_point.mode = 2
                    out_point.index = 3
                elif i == 2 and j == 4 and k == 6:
                    out_point.mode = 2
                    out_point.index = 39
                elif i == 2 and j == 5 and k == 2:
                    out_point.mode = 1
                    out_point.index = 38
                elif i == 2 and j == 5 and k == 3:
                    out_point.mode = 1
                    out_point.index = 2
                elif i == 2 and j == 5 and k == 5:
                    out_point.mode = 2
                    out_point.index = 2
                elif i == 2 and j == 5 and k == 6:
                    out_point.mode = 2
                    out_point.index = 38
                elif i == 2 and j == 6 and k == 2:
                    out_point.mode = 1
                    out_point.index = 37
                elif i == 2 and j == 6 and k == 3:
                    out_point.mode = 1
                    out_point.index = 1
                elif i == 2 and j == 6 and k == 5:
                    out_point.mode = 2
                    out_point.index = 1
                elif i == 2 and j == 6 and k == 6:
                    out_point.mode = 2
                    out_point.index = 37
                elif i == 2 and j == 7 and k == 2:
                    out_point.mode = 1
                    out_point.index = 36
                elif i == 2 and j == 7 and k == 3:
                    out_point.mode = 1
                    out_point.index = 0
                elif i == 2 and j == 7 and k == 5:
                    out_point.mode = 2
                    out_point.index = 0
                elif i == 2 and j == 7 and k == 6:
                    out_point.mode = 2
                    out_point.index = 36
                elif i == 3 and j == 2 and k == 2:
                    out_point.mode = 1
                    out_point.index = 42
                elif i == 3 and j == 2 and k == 3:
                    out_point.mode = 1
                    out_point.index = 6
                elif i == 3 and j == 2 and k == 5:
                    out_point.mode = 2
                    out_point.index = 6
                elif i == 3 and j == 2 and k == 6:
                    out_point.mode = 2
                    out_point.index = 42
                elif i == 3 and j == 3 and k == 2:
                    out_point.mode = 1
                    out_point.index = 59
                elif i == 3 and j == 3 and k == 3:
                    out_point.mode = 1
                    out_point.index = 23
                elif i == 3 and j == 3 and k == 5:
                    out_point.mode = 2
                    out_point.index = 23
                elif i == 3 and j == 3 and k == 6:
                    out_point.mode = 2
                    out_point.index = 59
                elif i == 3 and j == 4 and k == 2:
                    out_point.mode = 1
                    out_point.index = 58
                elif i == 3 and j == 4 and k == 3:
                    out_point.mode = 1
                    out_point.index = 22
                elif i == 3 and j == 4 and k == 5:
                    out_point.mode = 2
                    out_point.index = 22
                elif i == 3 and j == 4 and k == 6:
                    out_point.mode = 2
                    out_point.index = 58
                elif i == 3 and j == 5 and k == 2:
                    out_point.mode = 1
                    out_point.index = 57
                elif i == 3 and j == 5 and k == 3:
                    out_point.mode = 1
                    out_point.index = 21
                elif i == 3 and j == 5 and k == 5:
                    out_point.mode = 2
                    out_point.index = 21
                elif i == 3 and j == 5 and k == 6:
                    out_point.mode = 2
                    out_point.index = 57
                elif i == 3 and j == 6 and k == 2:
                    out_point.mode = 1
                    out_point.index = 56
                elif i == 3 and j == 6 and k == 3:
                    out_point.mode = 1
                    out_point.index = 20
                elif i == 3 and j == 6 and k == 5:
                    out_point.mode = 2
                    out_point.index = 20
                elif i == 3 and j == 6 and k == 6:
                    out_point.mode = 2
                    out_point.index = 56
                elif i == 3 and j == 7 and k == 2:
                    out_point.mode = 1
                    out_point.index = 55
                elif i == 3 and j == 7 and k == 3:
                    out_point.mode = 1
                    out_point.index = 19
                elif i == 3 and j == 7 and k == 5:
                    out_point.mode = 2
                    out_point.index = 19
                elif i == 3 and j == 7 and k == 6:
                    out_point.mode = 2
                    out_point.index = 55
                elif i == 4 and j == 2 and k == 2:
                    out_point.mode = 1
                    out_point.index = 43
                elif i == 4 and j == 2 and k == 3:
                    out_point.mode = 1
                    out_point.index = 7
                elif i == 4 and j == 2 and k == 5:
                    out_point.mode = 2
                    out_point.index = 7
                elif i == 4 and j == 2 and k == 6:
                    out_point.mode = 2
                    out_point.index = 43
                elif i == 4 and j == 3 and k == 2:
                    out_point.mode = 1
                    out_point.index = 60
                elif i == 4 and j == 3 and k == 3:
                    out_point.mode = 1
                    out_point.index = 24
                elif i == 4 and j == 3 and k == 5:
                    out_point.mode = 2
                    out_point.index = 24
                elif i == 4 and j == 3 and k == 6:
                    out_point.mode = 2
                    out_point.index = 60
                elif i == 4 and j == 4 and k == 2:
                    out_point.mode = 1
                    out_point.index = 69
                elif i == 4 and j == 4 and k == 3:
                    out_point.mode = 1
                    out_point.index = 33
                elif i == 4 and j == 4 and k == 5:
                    out_point.mode = 2
                    out_point.index = 33
                elif i == 4 and j == 4 and k == 6:
                    out_point.mode = 2
                    out_point.index = 69
                elif i == 4 and j == 5 and k == 2:
                    out_point.mode = 1
                    out_point.index = 68
                elif i == 4 and j == 5 and k == 3:
                    out_point.mode = 1
                    out_point.index = 32
                elif i == 4 and j == 5 and k == 5:
                    out_point.mode = 2
                    out_point.index = 32
                elif i == 4 and j == 5 and k == 6:
                    out_point.mode = 2
                    out_point.index = 68
                elif i == 4 and j == 6 and k == 2:
                    out_point.mode = 1
                    out_point.index = 67
                elif i == 4 and j == 6 and k == 3:
                    out_point.mode = 1
                    out_point.index = 31
                elif i == 4 and j == 6 and k == 5:
                    out_point.mode = 2
                    out_point.index = 31
                elif i == 4 and j == 6 and k == 6:
                    out_point.mode = 2
                    out_point.index = 67
                elif i == 4 and j == 7 and k == 2:
                    out_point.mode = 1
                    out_point.index = 54
                elif i == 4 and j == 7 and k == 3:
                    out_point.mode = 1
                    out_point.index = 18
                elif i == 4 and j == 7 and k == 5:
                    out_point.mode = 2
                    out_point.index = 18
                elif i == 4 and j == 7 and k == 6:
                    out_point.mode = 2
                    out_point.index = 54
                elif i == 5 and j == 2 and k == 2:
                    out_point.mode = 1
                    out_point.index = 44
                elif i == 5 and j == 2 and k == 3:
                    out_point.mode = 1
                    out_point.index = 8
                elif i == 5 and j == 2 and k == 5:
                    out_point.mode = 2
                    out_point.index = 8
                elif i == 5 and j == 2 and k == 6:
                    out_point.mode = 2
                    out_point.index = 44
                elif i == 5 and j == 3 and k == 2:
                    out_point.mode = 1
                    out_point.index = 61
                elif i == 5 and j == 3 and k == 3:
                    out_point.mode = 1
                    out_point.index = 25
                elif i == 5 and j == 3 and k == 5:
                    out_point.mode = 2
                    out_point.index = 25
                elif i == 5 and j == 3 and k == 6:
                    out_point.mode = 2
                    out_point.index = 61
                elif i == 5 and j == 4 and k == 2:
                    out_point.mode = 1
                    out_point.index = 70
                elif i == 5 and j == 4 and k == 3:
                    out_point.mode = 1
                    out_point.index = 34
                elif i == 5 and j == 4 and k == 5:
                    out_point.mode = 2
                    out_point.index = 34
                elif i == 5 and j == 4 and k == 6:
                    out_point.mode = 2
                    out_point.index = 70
                elif i == 5 and j == 5 and k == 2:
                    out_point.mode = 1
                    out_point.index = 71
                elif i == 5 and j == 5 and k == 3:
                    out_point.mode = 1
                    out_point.index = 35
                elif i == 5 and j == 5 and k == 5:
                    out_point.mode = 2
                    out_point.index = 35
                elif i == 5 and j == 5 and k == 6:
                    out_point.mode = 2
                    out_point.index = 71
                elif i == 5 and j == 6 and k == 2:
                    out_point.mode = 1
                    out_point.index = 66
                elif i == 5 and j == 6 and k == 3:
                    out_point.mode = 1
                    out_point.index = 30
                elif i == 5 and j == 6 and k == 5:
                    out_point.mode = 2
                    out_point.index = 30
                elif i == 5 and j == 6 and k == 6:
                    out_point.mode = 2
                    out_point.index = 66
                elif i == 5 and j == 7 and k == 2:
                    out_point.mode = 1
                    out_point.index = 53
                elif i == 5 and j == 7 and k == 3:
                    out_point.mode = 1
                    out_point.index = 17
                elif i == 5 and j == 7 and k == 5:
                    out_point.mode = 2
                    out_point.index = 17
                elif i == 5 and j == 7 and k == 6:
                    out_point.mode = 2
                    out_point.index = 53
                elif i == 6 and j == 2 and k == 2:
                    out_point.mode = 1
                    out_point.index = 45
                elif i == 6 and j == 2 and k == 3:
                    out_point.mode = 1
                    out_point.index = 9
                elif i == 6 and j == 2 and k == 5:
                    out_point.mode = 2
                    out_point.index = 9
                elif i == 6 and j == 2 and k == 6:
                    out_point.mode = 2
                    out_point.index = 45
                elif i == 6 and j == 3 and k == 2:
                    out_point.mode = 1
                    out_point.index = 62
                elif i == 6 and j == 3 and k == 3:
                    out_point.mode = 1
                    out_point.index = 26
                elif i == 6 and j == 3 and k == 5:
                    out_point.mode = 2
                    out_point.index = 26
                elif i == 6 and j == 3 and k == 6:
                    out_point.mode = 2
                    out_point.index = 62
                elif i == 6 and j == 4 and k == 2:
                    out_point.mode = 1
                    out_point.index = 63
                elif i == 6 and j == 4 and k == 3:
                    out_point.mode = 1
                    out_point.index = 27
                elif i == 6 and j == 4 and k == 5:
                    out_point.mode = 2
                    out_point.index = 27
                elif i == 6 and j == 4 and k == 6:
                    out_point.mode = 2
                    out_point.index = 63
                elif i == 6 and j == 5 and k == 2:
                    out_point.mode = 1
                    out_point.index = 64
                elif i == 6 and j == 5 and k == 3:
                    out_point.mode = 1
                    out_point.index = 28
                elif i == 6 and j == 5 and k == 5:
                    out_point.mode = 2
                    out_point.index = 28
                elif i == 6 and j == 5 and k == 6:
                    out_point.mode = 2
                    out_point.index = 64
                elif i == 6 and j == 6 and k == 2:
                    out_point.mode = 1
                    out_point.index = 65
                elif i == 6 and j == 6 and k == 3:
                    out_point.mode = 1
                    out_point.index = 29
                elif i == 6 and j == 6 and k == 5:
                    out_point.mode = 2
                    out_point.index = 29
                elif i == 6 and j == 6 and k == 6:
                    out_point.mode = 2
                    out_point.index = 65
                elif i == 6 and j == 7 and k == 2:
                    out_point.mode = 1
                    out_point.index = 52
                elif i == 6 and j == 7 and k == 3:
                    out_point.mode = 1
                    out_point.index = 16
                elif i == 6 and j == 7 and k == 5:
                    out_point.mode = 2
                    out_point.index = 16
                elif i == 6 and j == 7 and k == 6:
                    out_point.mode = 2
                    out_point.index = 52
                elif i == 7 and j == 2 and k == 2:
                    out_point.mode = 1
                    out_point.index = 46
                elif i == 7 and j == 2 and k == 3:
                    out_point.mode = 1
                    out_point.index = 10
                elif i == 7 and j == 2 and k == 5:
                    out_point.mode = 2
                    out_point.index = 10
                elif i == 7 and j == 2 and k == 6:
                    out_point.mode = 2
                    out_point.index = 46
                elif i == 7 and j == 3 and k == 2:
                    out_point.mode = 1
                    out_point.index = 47
                elif i == 7 and j == 3 and k == 3:
                    out_point.mode = 1
                    out_point.index = 11
                elif i == 7 and j == 3 and k == 5:
                    out_point.mode = 2
                    out_point.index = 11
                elif i == 7 and j == 3 and k == 6:
                    out_point.mode = 2
                    out_point.index = 47
                elif i == 7 and j == 4 and k == 2:
                    out_point.mode = 1
                    out_point.index = 48
                elif i == 7 and j == 4 and k == 3:
                    out_point.mode = 1
                    out_point.index = 12
                elif i == 7 and j == 4 and k == 5:
                    out_point.mode = 2
                    out_point.index = 12
                elif i == 7 and j == 4 and k == 6:
                    out_point.mode = 2
                    out_point.index = 48
                elif i == 7 and j == 5 and k == 2:
                    out_point.mode = 1
                    out_point.index = 49
                elif i == 7 and j == 5 and k == 3:
                    out_point.mode = 1
                    out_point.index = 13
                elif i == 7 and j == 5 and k == 5:
                    out_point.mode = 2
                    out_point.index = 13
                elif i == 7 and j == 5 and k == 6:
                    out_point.mode = 2
                    out_point.index = 49
                elif i == 7 and j == 6 and k == 2:
                    out_point.mode = 1
                    out_point.index = 50
                elif i == 7 and j == 6 and k == 3:
                    out_point.mode = 1
                    out_point.index = 14
                elif i == 7 and j == 6 and k == 5:
                    out_point.mode = 2
                    out_point.index = 14
                elif i == 7 and j == 6 and k == 6:
                    out_point.mode = 2
                    out_point.index = 50
                elif i == 7 and j == 7 and k == 2:
                    out_point.mode = 1
                    out_point.index = 51
                elif i == 7 and j == 7 and k == 3:
                    out_point.mode = 1
                    out_point.index = 15
                elif i == 7 and j == 7 and k == 5:
                    out_point.mode = 2
                    out_point.index = 15
                elif i == 7 and j == 7 and k == 6:
                    out_point.mode = 2
                    out_point.index = 51
                out_points.append(out_point)

    next_point = np.array(
        [
            [points[i].xyz[0], points[i].xyz[1], points[i].xyz[2]]
            for i in range(config.n_initial_points)
        ]
    )
    for i in range(config.n_initial_points):
        next_point = np.append(
            next_point,
            [[points[i].xyz[0], points[i].xyz[1], 0.05 - points[i].xyz[2]]],
            axis=0,
        )
    changes = Deformation.calculate_change(next_point)
    for i in range(config.n_prediction_points_init):
        if out_points[i].mode == 0:
            out_points[i].calculate_nextpoint(changes)
        elif out_points[i].mode == 1:
            out_points[i].xyz = points[out_points[i].index].xyz
        elif out_points[i].mode == 2:
            out_points[i].xyz = np.array(
                [
                    points[out_points[i].index].xyz[0],
                    points[out_points[i].index].xyz[1],
                    config.initial_coil_distance - points[out_points[i].index].xyz[2],
                ]
            )

    boxes = []
    edges_outside = []
    planes_outside = []
    edge_list = [
        [0, 1],
        [0, 2],
        [0, 4],
        [1, 3],
        [1, 5],
        [2, 3],
        [2, 6],
        [3, 7],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
    ]
    plane_list = [
        [0, 2, 3, 1],
        [0, 2, 6, 4],
        [0, 1, 5, 4],
        [1, 3, 7, 5],
        [2, 3, 7, 6],
        [4, 6, 7, 5],
    ]
    for i in range(9):
        for j in range(9):
            for k in range(8):
                if (k == 2 and i >= 2 and i <= 6 and j >= 2 and j <= 6) or (
                    k == 5 and i >= 2 and i <= 6 and j >= 2 and j <= 6
                ):
                    continue
                box = Box(
                    [
                        i * 90 + j * 9 + k,
                        i * 90 + j * 9 + (k + 1),
                        i * 90 + (j + 1) * 9 + k,
                        i * 90 + (j + 1) * 9 + k + 1,
                        (i + 1) * 90 + j * 9 + k,
                        (i + 1) * 90 + j * 9 + (k + 1),
                        (i + 1) * 90 + (j + 1) * 9 + k,
                        (i + 1) * 90 + (j + 1) * 9 + k + 1,
                    ]
                )
                index_for_plane = []
                for ii in range(8):
                    if (
                        out_points[box.node[ii]].mode == 1
                        or out_points[box.node[ii]].mode == 2
                    ):
                        index_for_plane.append(out_points[box.node[ii]].index)
                if len(index_for_plane) > 0 and len(index_for_plane) == 4:
                    flag = 0
                    for ii in range(len(planes)):
                        for jj in range(4):
                            for iii in range(4):
                                if planes[ii].node[iii] == index_for_plane[jj]:
                                    flag += 1
                                    break
                        if flag == 4:
                            box.plane = ii
                            break
                for edge_l in edge_list:
                    flag = 0
                    for edge_index in range(len(edges_outside)):
                        if (
                            edges_outside[edge_index].node[0] == box.node[edge_l[0]]
                            and edges_outside[edge_index].node[1] == box.node[edge_l[1]]
                        ):
                            flag = 1
                            box.edge_node.append(edge_index)
                            break
                    if flag == 0:
                        edge_outside = Edge_outside(
                            box.node[edge_l[0]], box.node[edge_l[1]]
                        )
                        box.edge_node.append(len(edges_outside))
                        edges_outside.append(edge_outside)

                for plane_l in plane_list:
                    for plane_index in range(len(planes_outside)):
                        flag = 0
                        for jj in range(4):
                            for iii in range(4):
                                if (
                                    planes_outside[plane_index].node[jj]
                                    == box.node[plane_l[iii]]
                                ):
                                    flag += 1
                                    break
                        if flag == 4:
                            box.plane_node.append(plane_index)
                            break
                    if flag != 4:
                        plane_outside = Plane_outside(
                            box.node[plane_l[0]],
                            box.node[plane_l[1]],
                            box.node[plane_l[2]],
                            box.node[plane_l[3]],
                        )
                        for edge_index in range(len(edges_outside)):
                            if (
                                edges_outside[edge_index].node[0]
                                == box.node[plane_l[0]]
                                and edges_outside[edge_index].node[1]
                                == box.node[plane_l[1]]
                            ) or (
                                edges_outside[edge_index].node[0]
                                == box.node[plane_l[1]]
                                and edges_outside[edge_index].node[1]
                                == box.node[plane_l[0]]
                            ):
                                plane_outside.edge[0] = edge_index
                            elif (
                                edges_outside[edge_index].node[0]
                                == box.node[plane_l[1]]
                                and edges_outside[edge_index].node[1]
                                == box.node[plane_l[2]]
                            ) or (
                                edges_outside[edge_index].node[0]
                                == box.node[plane_l[2]]
                                and edges_outside[edge_index].node[1]
                                == box.node[plane_l[1]]
                            ):
                                plane_outside.edge[1] = edge_index
                            elif (
                                edges_outside[edge_index].node[0]
                                == box.node[plane_l[2]]
                                and edges_outside[edge_index].node[1]
                                == box.node[plane_l[3]]
                            ) or (
                                edges_outside[edge_index].node[0]
                                == box.node[plane_l[3]]
                                and edges_outside[edge_index].node[1]
                                == box.node[plane_l[2]]
                            ):
                                plane_outside.edge[2] = edge_index
                            elif (
                                edges_outside[edge_index].node[0]
                                == box.node[plane_l[3]]
                                and edges_outside[edge_index].node[1]
                                == box.node[plane_l[0]]
                            ) or (
                                edges_outside[edge_index].node[0]
                                == box.node[plane_l[0]]
                                and edges_outside[edge_index].node[1]
                                == box.node[plane_l[3]]
                            ):
                                plane_outside.edge[3] = edge_index

                        box.plane_node.append(len(planes_outside))
                        planes_outside.append(plane_outside)
                boxes.append(box)
    return out_points, boxes, edges_outside, planes_outside
