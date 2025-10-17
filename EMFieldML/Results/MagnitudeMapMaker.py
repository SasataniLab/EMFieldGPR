"""Magnitude map generation for electromagnetic field visualization.

This module provides functionality to create magnitude maps and generate
training data for electromagnetic field prediction models.
"""

from pathlib import Path

import numpy as np

from EMFieldML.config import config, get_logger, paths, template

logger = get_logger(__name__)


class MakeYData:
    """Generate Y-data for magnitude map creation and training data preparation."""

    @staticmethod
    def make_ferrite_parameter():
        """Generate ferrite parameter data for magnitude map creation."""
        for y in range(25):
            for z in range(25):
                y_move = y * 0.01
                z_move = z * 0.01
                output_path = (
                    paths.RESULT_MAGNITUDE_MAP
                    / "circle_parameter"
                    / template.circle_move_param.format(index=y * 25 + z)
                )
                with Path(output_path).open("w") as f:
                    print(
                        0.14, 0.14, 0.13, 0.04, 0.01, 0.01, 0.01, y_move, z_move, file=f
                    )
                logger.info(f"Saved {output_path}")

    @staticmethod
    def make_prediction_point():
        """Generate prediction points for magnitude map visualization."""
        N = config.n_prediction_points_level2
        w = config.LBS_w_move

        # フェライトシールドの点
        with (
            paths.CIRCLE_MOVE_X_DIR / template.circle_move_x.format(index=1)
        ).open() as f:
            l_strip = [float(s.rstrip()) for s in f.readlines()]

        # TX(送信器)、RX(受信器)のフェライトシールドの点。それぞれ72点なのは基準点で、これらのみを元にlinearblendskinningを行うため
        TX_ferrite_point = np.array(
            [[0.0, 0.0, 0.0] for _ in range(config.n_initial_points)]
        )
        RX_ferrite_point = np.array(
            [[0.0, 0.0, 0.0] for _ in range(config.n_initial_points)]
        )
        for j in range(config.n_initial_points):
            TX_ferrite_point[j][0] = l_strip[j * 3]
            TX_ferrite_point[j][1] = l_strip[j * 3 + 1]
            TX_ferrite_point[j][2] = l_strip[j * 3 + 2]
            RX_ferrite_point[j][0] = l_strip[j * 3]
            RX_ferrite_point[j][1] = l_strip[j * 3 + 1]
            RX_ferrite_point[j][2] = config.initial_coil_distance - l_strip[j * 3 + 2]

        for y in range(25):
            for z in range(25):
                number = y * 25 + z

                # 移動する前の予測点
                prediction_point = np.loadtxt(
                    paths.CIRCLE_PREDICTION_POINT_DIR
                    / template.circle_prediction_point.format(index=1)
                )
                # 予測点それぞれに対してすべての基準点の重みを格納
                weight = np.zeros((N, config.n_initial_points * 2))
                for j in range(N):

                    weight_sum = 0  #

                    # それぞれのフェライトの点に対して重みを計算して合計する
                    for k in range(config.n_initial_points * 2):
                        if k < config.n_initial_points:
                            d = np.linalg.norm(
                                TX_ferrite_point[k] - prediction_point[j], ord=2
                            )
                            weight_sum += 1 / d ** (w)
                        else:
                            d = np.linalg.norm(
                                RX_ferrite_point[k - config.n_initial_points]
                                - prediction_point[j],
                                ord=2,
                            )
                            weight_sum += 1 / d ** (w)

                    # それぞれのフェライトの点に対する重みを正規化して、本当の重みとして保存する
                    for k in range(config.n_initial_points * 2):
                        if k < config.n_initial_points:
                            d = np.linalg.norm(
                                TX_ferrite_point[k] - prediction_point[j], ord=2
                            )
                            weight[j][k] = 1 / d ** (w) / weight_sum
                        else:
                            d = np.linalg.norm(
                                RX_ferrite_point[k - config.n_initial_points]
                                - prediction_point[j],
                                ord=2,
                            )
                            weight[j][k] = 1 / d ** (w) / weight_sum

                # フェライトシールドの各点の変化。RXのみy,zが動く
                change = np.array(
                    [[0.0, 0.0, 0.0] for _ in range(config.n_initial_points * 2)]
                )
                for k in range(config.n_initial_points, config.n_initial_points * 2):
                    change[k][1] = 0.01 * y
                    change[k][2] = 0.01 * z

                # prediction_pointにフェライトシールドの変化分を足す
                for j in range(N):
                    for k in range(config.n_initial_points * 2):
                        prediction_point[j] += change[k] * weight[j][k]

                output_path = (
                    paths.RESULT_MAGNITUDE_MAP
                    / "circle_prediction_point"
                    / template.circle_prediction_point_move.format(index=number)
                )
                # 最後に新しいprediction_pointを保存する
                with Path(output_path).open("w") as f:
                    for j in range(N):
                        print(
                            prediction_point[j][0],
                            prediction_point[j][1],
                            prediction_point[j][2],
                            file=f,
                        )

                logger.info(f"Saved {output_path}")
