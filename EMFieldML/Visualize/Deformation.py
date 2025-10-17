"""Coil deformation utilities for electromagnetic field visualization.

This module provides functions for deforming coils during electromagnetic
field analysis and visualization.


"""

import numpy as np

from EMFieldML.config import config, paths
from EMFieldML.Modeling import default_config


def change_X_test(X_test, TX_point):
    """Update X_test data with TX_point coordinates."""
    for i in range(len(TX_point)):
        for j in range(3):
            X_test[0][3 * i + j] = TX_point[i][j]


def change_h1(TX_point, RX_point, z_move, h1, h1_low, h1_middle):
    """Modify height parameter h1 for coil deformation."""
    for index in h1_low:
        TX_point[index][2] = -config.height_separate - h1
        RX_point[index][2] = (
            config.initial_coil_distance
            + config.height_separate
            + h1
            + z_move / config.meter_to_centimeter
        )

    for index in h1_middle:
        TX_point[index][2] = -config.height_separate - h1 / 2
        RX_point[index][2] = (
            config.initial_coil_distance
            + config.height_separate
            + h1 / 2
            + z_move / config.meter_to_centimeter
        )


def change_h2(TX_point, RX_point, z_move, h2, h2_list):
    """Modify height parameter h2 for coil deformation."""
    for index in h2_list:
        TX_point[index][2] = -config.height_separate + h2
        RX_point[index][2] = (
            config.initial_coil_distance
            + config.height_separate
            - h2
            + z_move / config.meter_to_centimeter
        )


def change_h3(TX_point, RX_point, z_move, h3, h3_list):
    """Modify height parameter h3 for coil deformation."""
    for index in h3_list:
        TX_point[index][2] = -config.height_separate + h3
        RX_point[index][2] = (
            config.initial_coil_distance
            + config.height_separate
            - h3
            + z_move / config.meter_to_centimeter
        )


def change_r1(TX_point, RX_point, y_move, r1, r1_list):
    """Modify radius parameter r1 for coil deformation."""
    now_r1 = (TX_point[r1_list[0]][0] ** 2 + TX_point[r1_list[0]][1] ** 2) ** (1 / 2)
    for index in r1_list:
        TX_point[index][0] *= r1 / now_r1
        TX_point[index][1] *= r1 / now_r1
        RX_point[index][0] *= r1 / now_r1
        RX_point[index][1] = (
            RX_point[index][1] - y_move / config.meter_to_centimeter
        ) * r1 / now_r1 + y_move / config.meter_to_centimeter


def change_r2(TX_point, RX_point, y_move, r2, r3, r2_list, r2_r3_list):
    """Modify radius parameter r2 for coil deformation."""
    now_r2 = (TX_point[r2_list[0]][0] ** 2 + TX_point[r2_list[0]][1] ** 2) ** (1 / 2)
    for index in r2_list:
        TX_point[index][0] *= r2 / now_r2
        TX_point[index][1] *= r2 / now_r2
        RX_point[index][0] *= r2 / now_r2
        RX_point[index][1] = (
            RX_point[index][1] - y_move / config.meter_to_centimeter
        ) * r2 / now_r2 + y_move / config.meter_to_centimeter
    for index in r2_r3_list:
        TX_point[index][0] *= (r2 + r3) / (now_r2 + r3)
        TX_point[index][1] *= (r2 + r3) / (now_r2 + r3)
        RX_point[index][0] *= (r2 + r3) / (now_r2 + r3)
        RX_point[index][1] = (
            RX_point[index][1] - y_move / config.meter_to_centimeter
        ) * (r2 + r3) / (now_r2 + r3) + y_move / config.meter_to_centimeter


def change_r3(TX_point, RX_point, y_move, r2, r3, r3_list, r2_r3_list):
    """Modify radius parameter r3 for coil deformation."""
    now_r3 = (TX_point[r3_list[0]][0] ** 2 + TX_point[r3_list[0]][1] ** 2) ** (1 / 2)
    for index in r3_list:
        TX_point[index][0] *= r3 / now_r3
        TX_point[index][1] *= r3 / now_r3
        RX_point[index][0] *= r3 / now_r3
        RX_point[index][1] = (
            RX_point[index][1] - y_move / config.meter_to_centimeter
        ) * r3 / now_r3 + y_move / config.meter_to_centimeter
    for index in r2_r3_list:
        TX_point[index][0] *= (r2 + r3) / (now_r3 + r2)
        TX_point[index][1] *= (r2 + r3) / (now_r3 + r2)
        RX_point[index][0] *= (r2 + r3) / (now_r3 + r2)
        RX_point[index][1] = (
            RX_point[index][1] - y_move / config.meter_to_centimeter
        ) * (r2 + r3) / (now_r3 + r2) + y_move / config.meter_to_centimeter


def change_r4(TX_point, RX_point, y_move, r4, r4_list):
    """Modify radius parameter r4 for coil deformation."""
    now_r4 = (TX_point[r4_list[4]][0] ** 2 + TX_point[r4_list[4]][1] ** 2) ** (
        1 / 2
    )  # r4はindex=0が端っこの点ではないので、index=4を使用
    for index in r4_list:
        TX_point[index][0] *= r4 / now_r4
        TX_point[index][1] *= r4 / now_r4
        RX_point[index][0] *= r4 / now_r4
        RX_point[index][1] = (
            RX_point[index][1] - y_move / config.meter_to_centimeter
        ) * r4 / now_r4 + y_move / config.meter_to_centimeter


def set_points(
    TX_point,
    TX_edge,
    TX_plane,
    RX_point,
    RX_edge,
    RX_plane,
    points_for_3D_TX,
    points_for_3D_RX,
):
    """Set point coordinates for coil deformation."""
    TX_point.update_point_positions(points_for_3D_TX)
    TX_edge.update_node_positions(points_for_3D_TX)
    TX_plane.update_vertex_positions(points_for_3D_TX)
    RX_point.update_point_positions(points_for_3D_RX)
    RX_edge.update_node_positions(points_for_3D_RX)
    RX_plane.update_vertex_positions(points_for_3D_RX)


def prepare_ferrite_move(points_for_3D_TX):
    """Prepare ferrite deformation parameters."""
    h1_low = []
    h1_middle = []
    h2_list = []
    h3_list = []
    r1_list = []
    r2_list = []
    r3_list = []
    r2_r3_list = []
    r4_list = []

    with (paths.DEFORMATION_DIR / "h1_low_point_index.txt").open() as f:
        for num in f:
            h1_low.append(int(num))
    with (paths.DEFORMATION_DIR / "h1_middle_point_index.txt").open() as f:
        for num in f:
            h1_middle.append(int(num))
    with (paths.DEFORMATION_DIR / "h2_point_index.txt").open() as f:
        for num in f:
            h2_list.append(int(num))
    with (paths.DEFORMATION_DIR / "h3_point_index.txt").open() as f:
        for num in f:
            h3_list.append(int(num))
    with (paths.DEFORMATION_DIR / "r1_point_index.txt").open() as f:
        for num in f:
            r1_list.append(int(num))
    with (paths.DEFORMATION_DIR / "r2_point_index.txt").open() as f:
        for num in f:
            r2_list.append(int(num))
    with (paths.DEFORMATION_DIR / "r3_point_index.txt").open() as f:
        for num in f:
            r3_list.append(int(num))
    with (paths.DEFORMATION_DIR / "r2_r3_middle_point_index.txt").open() as f:
        for num in f:
            r2_r3_list.append(int(num))
    with (paths.DEFORMATION_DIR / "r4_point_index.txt").open() as f:
        for num in f:
            r4_list.append(int(num))

    h1 = (
        abs(points_for_3D_TX[h1_low[0]][2] + config.height_separate)
        * config.meter_to_centimeter
    )
    h2 = (
        abs(points_for_3D_TX[h2_list[0]][2] + config.height_separate)
        * config.meter_to_centimeter
    )
    h3 = (
        abs(points_for_3D_TX[h3_list[0]][2] + config.height_separate)
        * config.meter_to_centimeter
    )
    r1 = (
        points_for_3D_TX[r1_list[0]][0] ** 2 + points_for_3D_TX[r1_list[0]][1] ** 2
    ) ** (1 / 2) * config.meter_to_centimeter
    r2 = (
        points_for_3D_TX[r2_list[0]][0] ** 2 + points_for_3D_TX[r2_list[0]][1] ** 2
    ) ** (1 / 2) * config.meter_to_centimeter
    r3 = (
        points_for_3D_TX[r3_list[0]][0] ** 2 + points_for_3D_TX[r3_list[0]][1] ** 2
    ) ** (1 / 2) * config.meter_to_centimeter
    r4 = (
        points_for_3D_TX[r4_list[4]][0] ** 2 + points_for_3D_TX[r4_list[4]][1] ** 2
    ) ** (
        1 / 2
    ) * config.meter_to_centimeter  # r4はindex=0が端っこの点ではないので、index=4を使用

    return (
        h1_low,
        h1_middle,
        h2_list,
        h3_list,
        r1_list,
        r2_list,
        r3_list,
        r2_r3_list,
        r4_list,
        h1,
        h2,
        h3,
        r1,
        r2,
        r3,
        r4,
    )


def move_out_points(
    out_points,
    points_for_3D_TX,
    y_move,
    z_move,
    points_for_3D_prediction_move,
    linearblend_skining_weight,
    points_for_3D_RX_original,
):
    """Move out points for coil deformation."""
    next_point = np.array(
        [
            [points_for_3D_TX[i][0], points_for_3D_TX[i][1], points_for_3D_TX[i][2]]
            for i in range(config.n_initial_points)
        ]
    )
    for i in range(config.n_initial_points):
        next_point = np.append(
            next_point,
            [
                [
                    points_for_3D_TX[i][0],
                    points_for_3D_TX[i][1],
                    config.initial_coil_distance - points_for_3D_TX[i][2],
                ]
            ],
            axis=0,
        )
    points_for_3D_prediction = np.empty((0, 3), float)
    changes = calculate_change(next_point)
    for i in range(len(out_points)):
        if i < config.n_prediction_points_init:
            if out_points[i].mode == 0:
                out_points[i].calculate_nextpoint(changes)
                points_for_3D_prediction = np.append(
                    points_for_3D_prediction, [out_points[i].xyz], axis=0
                )
            elif out_points[i].mode == 1:
                out_points[i].xyz = points_for_3D_TX[out_points[i].index]
            elif out_points[i].mode == 2:
                out_points[i].xyz = np.array(
                    [
                        points_for_3D_TX[out_points[i].index][0],
                        points_for_3D_TX[out_points[i].index][1],
                        config.initial_coil_distance
                        - points_for_3D_TX[out_points[i].index][2],
                    ]
                )
        elif out_points[i].mode == 0:
            new_point = np.zeros(3)
            for j in out_points[i].parent:
                new_point += out_points[j].xyz
            out_points[i].xyz = new_point / len(out_points[i].parent)
            points_for_3D_prediction = np.append(
                points_for_3D_prediction, [out_points[i].xyz], axis=0
            )
        elif out_points[i].mode == 1:
            out_points[i].xyz = points_for_3D_TX[out_points[i].index]
        elif out_points[i].mode == 2:
            out_points[i].xyz = np.array(
                [
                    points_for_3D_TX[out_points[i].index][0],
                    points_for_3D_TX[out_points[i].index][1],
                    config.initial_coil_distance
                    - points_for_3D_TX[out_points[i].index][2],
                ]
            )

    change_yz = np.array([[0.0, 0.0, 0.0] for _ in range(config.n_initial_points * 2)])
    for k in range(config.n_initial_points):
        change_yz[k + config.n_initial_points] = np.array(
            [
                0.0,
                y_move / config.meter_to_centimeter,
                z_move / config.meter_to_centimeter,
            ]
        )
    for i in range(len(points_for_3D_prediction_move)):
        points_for_3D_prediction_move[i] = points_for_3D_prediction[i] + np.dot(
            linearblend_skining_weight[i], change_yz
        )

    for i in range(len(points_for_3D_TX)):
        points_for_3D_RX_original[i] = np.array(
            [
                points_for_3D_TX[i][0],
                points_for_3D_TX[i][1],
                config.initial_coil_distance - points_for_3D_TX[i][2],
            ]
        )


def calculate_change(next_point):
    """Calculate change in point positions for coil deformation."""
    points_default = default_config.points_default
    changes = []
    for i in range(config.n_initial_points // 2):
        change = next_point[i] - points_default[i]
        changes.append(change)
    for i in range(config.n_initial_points // 2):
        change = next_point[config.n_initial_points // 2 + i] - np.array(
            [points_default[i][0], points_default[i][1], -0.06]
        )
        changes.append(change)
    for i in range(config.n_initial_points // 2):
        change = next_point[config.n_initial_points + i] - np.array(
            [points_default[i][0], points_default[i][1], 0.05]
        )
        changes.append(change)
    for i in range(config.n_initial_points // 2):
        change = next_point[config.n_initial_points // 2 * 3 + i] - np.array(
            [points_default[i][0], points_default[i][1], 0.11]
        )
        changes.append(change)
    return np.array(changes)
