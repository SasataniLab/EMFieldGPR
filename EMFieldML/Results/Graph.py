"""Graph generation and visualization module for EMFieldML results."""

import csv
import math
import statistics
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

try:
    import pandas as pd
except (ImportError, TypeError):
    # Handle both import errors and version compatibility issues
    pd = None
import seaborn as sns
from matplotlib.pyplot import figure

from EMFieldML.config import config, paths, template
from EMFieldML.Visualize import Visualize


class GraphMaker:
    """Class for generating various graphs and visualizations of EMFieldML results."""

    @staticmethod
    def n_change_deviation() -> None:
        """Generate deviation graph for N change analysis."""
        number_list = config.selected_points
        data_set = []
        for num in number_list:
            data = np.loadtxt(
                paths.RESULT_N_CHANGE_DIR
                / template.result_n_change_deviation_sorted.format(index=num)
            )
            data_set.append(data)
        data_set = np.array(data_set)
        n_list = [
            config.n_initial_train_data + i * config.n_initial_train_data // 6
            for i in range(7)
        ]
        ## センチメートル単位でフィギュアサイズを設定します
        width_cm = 4
        height_cm = 3
        ## cmをinchに変換
        width_in = width_cm / 2.54
        height_in = height_cm / 2.54
        ## 指定されたサイズで新しいフィギュアを作成します
        figure(figsize=(width_in, height_in))

        plt.plot(
            n_list,
            [data_set[0][i][1] for i in range(7)],
            label="1",
            marker="s",
            markersize=4,
            ls=(0, (1, 1)),
        )
        plt.plot(
            n_list,
            [data_set[1][i][1] for i in range(7)],
            label="2",
            marker="o",
            markersize=4,
            ls=(0, (2, 1)),
        )
        plt.plot(
            n_list,
            [data_set[2][i][1] for i in range(7)],
            label="3",
            marker="v",
            markersize=4,
            ls=(0, (5, 1)),
        )
        plt.plot(
            n_list,
            [data_set[3][i][1] for i in range(7)],
            label="4",
            marker="^",
            markersize=4,
            ls=(0, (3, 1, 1, 1)),
        )
        plt.plot(
            n_list,
            [data_set[4][i][1] for i in range(7)],
            label="5",
            marker="*",
            markersize=4,
            ls=(0, (5, 1, 1, 1)),
        )
        plt.plot(
            n_list,
            [data_set[5][i][1] for i in range(7)],
            label="6",
            marker="x",
            markersize=4,
            ls=(0, (5, 3, 1, 3, 1, 3)),
        )

        plt.xlabel("Number of training samples Nt", fontsize=7)
        plt.ylabel("Max deviation", fontsize=7)

        plt.xlim(700, 1532)
        plt.ylim(0.86, 0.92)

        plt.xticks(
            [
                config.n_initial_train_data + i * config.n_initial_train_data // 3
                for i in range(4)
            ],
            fontsize=6,
        )
        plt.yticks([0.86, 0.88, 0.90], fontsize=6)

        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        plt.grid(axis="y", linewidth=1)
        plt.legend(ncol=2, fontsize=6, columnspacing=1, borderaxespad=0)
        # Display plot (apply settings)
        plt.show()

    @staticmethod
    def n_change_error() -> None:
        """Generate error graph for N change analysis."""
        number_list = config.selected_points
        data_set = []
        for num in number_list:
            data = np.loadtxt(
                paths.RESULT_N_CHANGE_DIR
                / template.result_n_change_error_sorted.format(index=num)
            )
            data_set.append(data)
        data_set = np.array(data_set)
        n_list = [
            config.n_initial_train_data + i * config.n_initial_train_data // 6
            for i in range(7)
        ]
        ## センチメートル単位でフィギュアサイズを設定します
        width_cm = 4
        height_cm = 3
        ## cmをinchに変換
        width_in = width_cm / 2.54
        height_in = height_cm / 2.54
        ## 指定されたサイズで新しいフィギュアを作成します
        figure(figsize=(width_in, height_in))

        plt.plot(
            n_list,
            [data_set[0][i][1] for i in range(7)],
            label="1",
            marker="s",
            markersize=4,
            ls=(0, (1, 1)),
        )
        plt.plot(
            n_list,
            [data_set[1][i][1] for i in range(7)],
            label="2",
            marker="o",
            markersize=4,
            ls=(0, (2, 1)),
        )
        plt.plot(
            n_list,
            [data_set[2][i][1] for i in range(7)],
            label="3",
            marker="v",
            markersize=4,
            ls=(0, (5, 1)),
        )
        plt.plot(
            n_list,
            [data_set[3][i][1] for i in range(7)],
            label="4",
            marker="^",
            markersize=4,
            ls=(0, (3, 1, 1, 1)),
        )
        plt.plot(
            n_list,
            [data_set[4][i][1] for i in range(7)],
            label="5",
            marker="*",
            markersize=4,
            ls=(0, (5, 1, 1, 1)),
        )
        plt.plot(
            n_list,
            [data_set[5][i][1] for i in range(7)],
            label="6",
            marker="x",
            markersize=4,
            ls=(0, (5, 3, 1, 3, 1, 3)),
        )

        plt.xlabel("Number of training samples Nt", fontsize=7)
        plt.ylabel("Max error", fontsize=7)

        plt.xlim(700, 1532)
        plt.ylim(0, 4.5)

        plt.xticks(
            [
                config.n_initial_train_data + i * config.n_initial_train_data // 3
                for i in range(4)
            ],
            fontsize=6,
        )
        plt.yticks(
            [0, 1, 2, 3, 4], ["0.00", "1.00", "2.00", "3.00", "4.00"], fontsize=6
        )

        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        plt.grid(axis="y", linewidth=1)
        plt.legend(ncol=2, fontsize=6, columnspacing=1, borderaxespad=0)
        # Display plot (apply settings)
        plt.show()

    @staticmethod
    def active_learning_map(
        input_path: Path,
        n_data: int,
    ):
        """Generate active learning map visualization."""
        with Path(input_path).open() as f:
            lines = f.readlines()
            x_train = [int(line.strip()) for line in lines]

        x_ans = []
        y_ans = []

        for i in range(n_data):
            number = x_train[i] - 1
            num = number // config.n_shield_shape
            x = (num // 13) * 2
            y = (num % 13) * 2
            x_ans.append(x)
            y_ans.append(y)

        bins = [13, 13]
        xrange = [0, 25]
        yrange = [0, 25]

        (bins[0]) * (bins[1])

        hist, edgesx, edgesy = np.histogram2d(
            x_ans, y_ans, bins=bins, range=[xrange, yrange]
        )
        hist = np.flipud(np.rot90(hist))  # histogram2d 出力の仕様由来のおまじない

        xpos, ypos = np.meshgrid(edgesx[:-1], edgesy[:-1])

        ## センチメートル単位でフィギュアサイズを設定します
        width_cm = 3.5
        height_cm = 3
        ## cmをinchに変換
        width_in = width_cm / 2.54
        height_in = height_cm / 2.54
        ## 指定されたサイズで新しいフィギュアを作成します
        figure(figsize=(width_in, height_in))

        plt.pcolormesh(
            xpos, ypos, hist, cmap="RdYlBu_r"
        )  # 等高線図の生成。cmapで色付けの規則を指定する。
        plt.clim(0, 39)

        pp = plt.colorbar(orientation="vertical")  # カラーバーの表示
        pp.set_label(
            "Number of samples", fontname="Helvetica", fontsize=7
        )  # カラーバーのラベル
        pp.ax.tick_params(labelsize=7)

        plt.xlabel("x (cm)", fontsize=7)
        plt.ylabel("z (cm)", fontsize=7)

        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)

        plt.show()

    @staticmethod
    def magnitude_adaptive() -> None:
        """Generate adaptive magnitude visualization."""
        ## センチメートル単位でフィギュアサイズを設定します
        width_cm = 4
        height_cm = 4
        ## cmをinchに変換
        width_in = width_cm / 2.54
        height_in = height_cm / 2.54
        fix_list = []
        adaptive_list = []
        with (
            paths.RESULT_CROSS_DIR / template.result_predict_magnitude_no_move
        ).open() as f:
            for num in f:
                fix_list.append(float(num))
        with (
            paths.RESULT_CROSS_DIR / template.result_predict_magnitude_move
        ).open() as f:
            for num in f:
                adaptive_list.append(float(num))

        figure(figsize=(width_in, height_in))
        plt.rcParams["font.family"] = "Helvetica"
        plt.rcParams["font.size"] = 7
        plt.rcParams["axes.axisbelow"] = True
        plt.grid(axis="y", linewidth=0.5)
        plt.hist(
            adaptive_list,
            bins=[10 ** (-1.4 + i * 0.08) for i in range(86)],
            color="blue",
            alpha=0.5,
            label="Adaptive Exterior Grid",
        )
        plt.hist(
            fix_list,
            bins=[10 ** (0.6 + i * 0.08) for i in range(73)],
            color="red",
            alpha=0.5,
            label="Fixed Grid",
        )
        plt.xlabel("MSE in magnetic field magnitude", fontsize=7)
        plt.ylabel("Number of samples", fontsize=7)
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)
        plt.xscale("log")
        plt.ylim(0, 80)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1))
        plt.show()

    @staticmethod
    def vector_adaptive() -> None:
        """Generate adaptive vector visualization."""
        ## センチメートル単位でフィギュアサイズを設定します
        width_cm = 4
        height_cm = 4
        ## cmをinchに変換
        width_in = width_cm / 2.54
        height_in = height_cm / 2.54
        fix_list = []
        adaptive_list = []
        with (
            paths.RESULT_CROSS_DIR / template.result_predict_vector_no_move
        ).open() as f:
            for num in f:
                fix_list.append(float(num))
        with (paths.RESULT_CROSS_DIR / template.result_predict_vector_move).open() as f:
            for num in f:
                adaptive_list.append(float(num))

        ## 指定されたサイズで新しいフィギュアを作成します
        figure(figsize=(width_in, height_in))
        plt.rcParams["font.family"] = "Helvetica"
        plt.rcParams["font.size"] = 7
        plt.rcParams["axes.axisbelow"] = True
        plt.grid(axis="y", linewidth=0.5)
        plt.hist(
            adaptive_list,
            bins=list(range(150)),
            color="blue",
            alpha=0.5,
            label="Adaptive Exterior Grid",
        )
        plt.hist(
            fix_list, bins=list(range(150)), color="red", alpha=0.5, label="Fixed Grid"
        )
        plt.xlabel("Absolute error in vector angle (deg)", fontsize=7)
        plt.ylabel("Number of samples", fontsize=7)
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)
        plt.ylim(0, 110)
        plt.yticks(list(range(0, 110, 20)))
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1))
        plt.show()

    @staticmethod
    def magnitude_postprocessing_compare() -> None:
        """Compare magnitude postprocessing results."""
        data = []
        n = 20
        for i in range(1, n + 1):
            data_now = []
            with (
                paths.RESULT_FULL_DATA_DIR
                / template.result_postprocessing_change_magnitude_error.format(index=i)
            ).open() as f:
                for mag in f:
                    data_now.append(float(mag))
            data.append(data_now)

        width_cm = 5
        height_cm = 4
        ## cmをinchに変換
        width_in = width_cm / 2.54
        height_in = height_cm / 2.54
        ## 指定されたサイズで新しいフィギュアを作成します
        fig = plt.figure(figsize=(width_in, height_in))
        ax1 = fig.add_subplot(111)

        for i in range(n):
            if i == 0:
                ax1.scatter(
                    [i + 1] * len(data[i]),
                    data[i],
                    marker=".",
                    s=20,
                    color="skyblue",
                    label="error",
                )
            else:
                ax1.scatter(
                    [i + 1] * len(data[i]), data[i], marker=".", s=20, color="skyblue"
                )

        pstdev = []
        for i in range(n):
            pstdev.append(statistics.pstdev(data[i]))

        x = [i + 1 for i in range(n)]
        ax1.plot(
            x,
            [np.mean(data[i]) for i in range(n)],
            marker="^",
            markersize=5,
            color="gray",
            label="error average",
        )

        ax2 = ax1.twinx()
        ax2.plot(
            x,
            pstdev,
            marker="x",
            markersize=7,
            color="orange",
            label="standard deviation",
        )
        ax1.set_xlabel("s", fontsize=7)
        ax1.set_ylabel("Relative error in magnitude", fontsize=7)
        ax2.set_ylabel("Standard deviation", fontsize=7)
        ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax1.set_yticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)
        ax1.set_xticks([5, 10, 15, 20])
        ax1.set_xticklabels([5, 10, 15, 20], fontsize=7)
        ax2.set_yscale("log")
        ax2.set_yticks([0.01, 0.05, 0.1, 0.5])
        ax2.set_yticklabels(["1e-2", "5e-2", "1e-1", "5e-1"], fontsize=7)
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(
            h1 + h2, l1 + l2, loc="lower center", bbox_to_anchor=(0.5, 1), fontsize=7
        )
        plt.show()

    @staticmethod
    def efficiency_postprocessing_compare() -> None:
        """Compare efficiency postprocessing results."""
        data = []

        for i in [10, 8, 6, 4, 2]:
            data_now = []
            with (
                paths.RESULT_FULL_DATA_DIR
                / template.result_postprocessing_change_efficiency_error_under.format(
                    index=i
                )
            ).open() as f:
                for mag in f:
                    data_now.append(float(mag))
            data.append(data_now)

        for i in [1, 2, 3, 5, 10]:
            data_now = []
            with (
                paths.RESULT_FULL_DATA_DIR
                / template.result_postprocessing_change_efficiency_error.format(index=i)
            ).open() as f:
                for mag in f:
                    data_now.append(float(mag))
            data.append(data_now)

        width_cm = 7
        height_cm = 6
        ## cmをinchに変換
        width_in = width_cm / 2.54
        height_in = height_cm / 2.54
        ## 指定されたサイズで新しいフィギュアを作成します
        fig = plt.figure(figsize=(width_in, height_in))
        ax1 = fig.add_subplot(111)

        ax1.scatter(
            [1 / 10] * len(data[0]),
            data[0],
            marker=".",
            s=20,
            color="skyblue",
            label="error",
        )
        ax1.scatter([1 / 8] * len(data[1]), data[1], marker=".", s=20, color="skyblue")
        ax1.scatter([1 / 6] * len(data[2]), data[2], marker=".", s=20, color="skyblue")
        ax1.scatter([1 / 4] * len(data[3]), data[3], marker=".", s=20, color="skyblue")
        ax1.scatter([1 / 2] * len(data[4]), data[4], marker=".", s=20, color="skyblue")
        ax1.scatter([1] * len(data[5]), data[5], marker=".", s=20, color="skyblue")
        ax1.scatter([2] * len(data[6]), data[6], marker=".", s=20, color="skyblue")
        ax1.scatter([3] * len(data[7]), data[7], marker=".", s=20, color="skyblue")
        ax1.scatter([5] * len(data[8]), data[8], marker=".", s=20, color="skyblue")
        ax1.scatter([10] * len(data[9]), data[9], marker=".", s=20, color="skyblue")

        pstdev = []
        for i in range(10):
            pstdev.append(statistics.pstdev(data[i]))

        x = [1 / 10, 1 / 8, 1 / 6, 1 / 4, 1 / 2, 1, 2, 3, 5, 10]
        ax1.plot(
            x,
            [np.mean(data[i]) for i in range(10)],
            marker="^",
            markersize=5,
            color="gray",
            label="error average",
        )

        ax2 = ax1.twinx()

        ax2.plot(
            x,
            pstdev,
            marker="x",
            markersize=5,
            color="orange",
            label="standard deviation",
        )

        ax1.set_xlabel("s", fontsize=7)
        ax1.set_ylabel("Relative error in efficiency", fontsize=7)
        ax2.set_ylabel("Standard deviation", fontsize=7)

        ax1.set_yscale("log")
        ax1.set_yticks([0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1])
        ax1.set_yticklabels(
            ["1e-6", "1e-5", "1e-4", "1e-3", "1e-2", "1e-1", "1"], fontsize=7
        )

        ax1.set_xticks([5, 10, 15, 20])
        ax1.set_xticklabels([5, 10, 15, 20], fontsize=7)

        ax1.set_xscale("log")

        ax2.set_yscale("log")
        ax2.set_yticks([0.0001, 0.001, 0.01, 0.1])
        ax2.set_yticklabels(["1e-4", "1e-3", "1e-2", "1e-1"], fontsize=7)

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(
            h1 + h2, l1 + l2, loc="lower center", bbox_to_anchor=(0.5, 1), fontsize=7
        )
        plt.show()

    @staticmethod
    def error_map(
        input_path: Path,
        type_error: str,
        vmax: float,
        cmap: str = "Reds",
    ):
        """Generate error map visualization."""
        error = []
        with Path(input_path).open() as f:
            for mag in f:
                error.append(float(mag))
        position = []
        for i in range(1, config.n_test_data + 1):
            with (
                paths.TEST_CIRCLE_MOVE_X_DIR / template.test_circle_move.format(index=i)
            ).open() as f:
                data = f.readlines()
                coordinate_data = [[float(row.rstrip("\n")) for row in data]][0]  #
            position.append([coordinate_data[-2] * 100, coordinate_data[-1] * 100])
        position = np.array(position)
        error = np.array(error)
        width_cm = 5
        height_cm = 4
        ## cmをinchに変換
        width_in = width_cm / 2.54
        height_in = height_cm / 2.54
        ## 指定されたサイズで新しいフィギュアを作成します
        figure(figsize=(width_in, height_in))
        sc = plt.scatter(
            position[:, 0], position[:, 1], vmin=0, vmax=vmax, c=error, cmap=cmap, s=20
        )
        pp = plt.colorbar(sc)
        if type_error == "vector":
            pp.set_label(
                "angle error", fontname="Helvetica", fontsize=7
            )  # カラーバーのラベル
        else:
            pp.set_label(
                f"Relative error in {type_error}", fontname="Helvetica", fontsize=7
            )
        pp.ax.tick_params(labelsize=7)
        plt.xlabel("y (cm)", fontsize=7)
        plt.ylabel("z (cm)", fontsize=7)
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)
        plt.show()

    @staticmethod
    def magnitude_map_efficiency(
        postprocessing: float,
    ):
        """Generate magnitude map with efficiency analysis."""
        ## センチメートル単位でフィギュアサイズを設定します
        width_cm = 5.5
        height_cm = 4.5
        ## cmをinchに変換
        width_in = width_cm / 2.54
        height_in = height_cm / 2.54

        # 描画エリアの作成
        fig = plt.figure(figsize=(width_in, height_in))
        ax = fig.add_subplot(111, projection="3d")
        font = 7
        # 各軸の設定
        ax.set_xlabel("y (cm)", size=font, labelpad=-9)
        ax.set_ylabel("z (cm)", size=font, labelpad=-9)
        ax.set_zlabel("Efficiency", size=font, labelpad=-9)
        ax.set_xlim(0, 24)
        ax.set_ylim(0, 24)
        ax.set_zlim(0, 1)
        ax.set_xticks([0, 10, 20])
        ax.set_xticklabels(["0", "10", "20"], fontsize=font)
        ax.set_yticks([0, 10, 20])
        ax.set_yticklabels(["0", "10", "20"], fontsize=font)
        ax.set_zticks([0, 0.5, 1.0])
        ax.set_zticklabels(["0.0", "0.5", "1.0"], fontsize=font)
        ax.tick_params(pad=-3.5)
        # x軸、y軸の作成
        x = np.array([i * 1.0 for i in range(25)])
        y = np.array([i * 1.0 for i in range(25)])
        # メッシュの作成
        x_mesh, y_mesh = np.meshgrid(x, y)

        # 磁界分布の作成
        z_mesh = x_mesh**2
        with (
            paths.RESULT_MAGNITUDE_MAP_DIR / paths.Y_DATA / template.y_data_efficiency
        ).open() as f:
            reader = csv.reader(f)
            csv_data = list(reader)  #
            for i in range(25):
                for j in range(25):
                    z_mesh[j][i] = float(csv_data[i * 25 + j][0]) ** (
                        1 / postprocessing
                    )

        # 曲面の描画
        ax.plot_surface(x_mesh, y_mesh, z_mesh, cmap="coolwarm")
        ax.view_init(elev=16, azim=-116)
        fig.subplots_adjust(left=0, right=1, bottom=0.1, top=1)

        plt.show()

    @staticmethod
    def magnitude_map_magnetic(
        postprocessing: int,
    ):
        """Generate magnitude map with magnetic field analysis."""
        ## センチメートル単位でフィギュアサイズを設定します
        width_cm = 5.5
        height_cm = 4.5
        ## cmをinchに変換
        width_in = width_cm / 2.54
        height_in = height_cm / 2.54

        # 描画エリアの作成
        fig = plt.figure(figsize=(width_in, height_in))
        ax = fig.add_subplot(111, projection="3d")
        font = 7
        # 各軸の設定
        ax.set_xlabel("y (cm)", size=font, labelpad=-9)
        ax.set_ylabel("z (cm)", size=font, labelpad=-9)
        ax.set_zlabel("Magnetic field (A/m)", size=font, labelpad=-9)
        ax.set_xlim(0, 24)
        ax.set_ylim(0, 24)
        if postprocessing == 1:
            ax.set_zlim(0, 24)
            ax.set_zticks([0, 10, 20])
            ax.set_zticklabels(["0", "10", "20"], fontsize=font)
        elif postprocessing >= 2:
            ax.set_zlim(0.8, 2.0)
            ax.set_zticks([1, 1.5, 2])
            ax.set_zticklabels(["1.0", "1.5", "2.0"], fontsize=font)
        ax.set_xticks([0, 10, 20])
        ax.set_xticklabels(["0", "10", "20"], fontsize=font)
        ax.set_yticks([0, 10, 20])
        ax.set_yticklabels(["0", "10", "20"], fontsize=font)
        ax.tick_params(pad=-3.5)
        # x軸、y軸の作成
        x = np.array([i * 1.0 for i in range(25)])
        y = np.array([i * 1.0 for i in range(25)])
        # メッシュの作成
        x_mesh, y_mesh = np.meshgrid(x, y)

        # 磁界分布の作成
        z_mesh = x_mesh**2
        with (
            paths.RESULT_MAGNITUDE_MAP_DIR
            / paths.Y_DATA
            / template.y_data_magnitude.format(index=0)
        ).open() as f:
            reader = csv.reader(f)
            csv_data = list(reader)  #
            for i in range(25):
                for j in range(25):
                    z_mesh[j][i] = float(csv_data[i * 25 + j][0]) ** (
                        1 / postprocessing
                    )

        # 曲面の描画
        ax.plot_surface(x_mesh, y_mesh, z_mesh, cmap="coolwarm")
        ax.view_init(elev=16, azim=-116)
        fig.subplots_adjust(left=0, right=1, bottom=0.1, top=1)

        plt.show()

    @staticmethod
    def change_emfield_by_roi_and_movement(aimpoint, vmax, vmin, relative_move=False):
        """Change EM field by ROI and movement analysis."""
        shield_number = 90
        visualize = Visualize(shield_number)
        visualize.z_move = 20.0
        max_value = 0.0
        min_value = 1000
        segments = 50
        template_aimpoint = aimpoint.copy()

        magnitude_list = np.array([])
        for i in range(segments + 1):
            visualize.r1 = (
                visualize.r1_min + i * (visualize.r1_max - visualize.r1_min) / segments
            )
            visualize.update_values_for_shape(
                changed_h1=False,
                changed_h2=False,
                changed_h3=False,
                changed_r1=True,
                changed_r2=False,
                changed_r3=False,
                changed_r4=False,
            )
            for j in range(segments + 1):
                visualize.y_move = j * 20.0 / segments
                visualize.update_values_for_move()
                values = visualize.get_values()
                points = visualize.get_prediction_points()
                if relative_move:
                    aimpoint = [
                        template_aimpoint[0],
                        template_aimpoint[1] + visualize.y_move / 100,
                        template_aimpoint[2],
                    ]
                based_point = [
                    GraphMaker.search_near_point(aimpoint, points, [-1, -1, -1]),
                    GraphMaker.search_near_point(aimpoint, points, [1, -1, -1]),
                    GraphMaker.search_near_point(aimpoint, points, [-1, 1, -1]),
                    GraphMaker.search_near_point(aimpoint, points, [1, 1, -1]),
                    GraphMaker.search_near_point(aimpoint, points, [-1, -1, 1]),
                    GraphMaker.search_near_point(aimpoint, points, [1, -1, 1]),
                    GraphMaker.search_near_point(aimpoint, points, [-1, 1, 1]),
                    GraphMaker.search_near_point(aimpoint, points, [1, 1, 1]),
                ]
                all_weight = 0.0
                magnitude = 0.0
                for k in based_point:
                    weight = 1 / math.sqrt(
                        (points[k][0] - aimpoint[0]) ** 2
                        + (points[k][1] - aimpoint[1]) ** 2
                        + (points[k][2] - aimpoint[2]) ** 2
                    )
                    all_weight += weight
                    magnitude += values[k] * weight
                magnitude /= all_weight
                magnitude_list = np.append(magnitude_list, magnitude)
                max_value = max(max_value, magnitude)
                min_value = min(min_value, magnitude)

        print("Max value: ", max_value)
        print("Min value: ", min_value)
        magnitude_list = magnitude_list.reshape((segments + 1, segments + 1))
        if pd is None:
            raise ImportError("pandas is required for this functionality")
        magnitude_df = pd.DataFrame(data=magnitude_list)
        ## センチメートル単位でフィギュアサイズを設定します
        width_cm = 4.5
        height_cm = 4.0
        ## cmをinchに変換
        width_in = width_cm / 2.54
        height_in = height_cm / 2.54
        ## 指定されたサイズで新しいフィギュアを作成します
        fig, ax = plt.subplots(figsize=(width_in, height_in))

        cmthermal = GraphMaker.generate_mycmap()
        im = sns.heatmap(
            magnitude_df, square=True, cmap=cmthermal, vmax=vmax, vmin=vmin, cbar=False
        )
        cbar = plt.colorbar(im.collections[0], ax=ax, shrink=0.8, pad=0.1)
        cax = cbar.ax
        sns.despine()
        ax.invert_yaxis()
        ax.collections[0].colorbar.minorticks_on()
        cax = ax.collections[0].colorbar.ax
        cax.tick_params(which="minor")
        cax.tick_params(which="major", labelsize=7)
        # ax.collections[0].colorbar.set_label(
        #     "Magnetic field (A/m)", fontsize=7
        # )  # カラーバーのラベル

        # 軸ラベルを設定
        tick_positions = np.arange(0, segments + 1, 25)
        # ラベルを作成
        x_labels = []
        y_labels = []

        # h2とh3の値を計算し、小数点第一位に丸めてリストに追加
        for i in range(len(tick_positions)):
            y_value = i * 10.0
            r1_value = visualize.r1_min + i * 6.0

            # 小数点第一位に丸めて、f-stringで文字列化
            x_labels.append(f"{y_value:.1f}")
            y_labels.append(f"{r1_value:.1f}")

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(x_labels, rotation=45, fontsize=7)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(y_labels, fontsize=7)

        plt.xlabel(r"position($\it{y}$) (cm)", fontsize=7)
        plt.ylabel(r"$\it{r}_1$ (cm)", fontsize=7)
        plt.tight_layout()
        plt.savefig(
            "changeEMfieldByRoiAndMovementFixPointPointB.jpg", format="jpg", dpi=500
        )
        plt.show()

    @staticmethod
    def change_efficiency_by_roi_and_movement() -> None:
        """Change efficiency by ROI and movement analysis."""
        shield_number = 90
        visualize = Visualize(shield_number)
        visualize.z_move = 20.0
        segments = 50

        efficiency_list = np.array([])
        for i in range(segments + 1):
            visualize.r1 = (
                visualize.r1_min + i * (visualize.r1_max - visualize.r1_min) / segments
            )
            visualize.update_values_for_shape(
                changed_h1=False,
                changed_h2=False,
                changed_h3=False,
                changed_r1=True,
                changed_r2=False,
                changed_r3=False,
                changed_r4=False,
            )
            for j in range(segments + 1):
                visualize.y_move = j * 20.0 / segments
                visualize.update_values_for_move()
                efficiency = visualize.get_efficiency() * 100
                efficiency_list = np.append(efficiency_list, efficiency)

        efficiency_list = efficiency_list.reshape((segments + 1, segments + 1))
        if pd is None:
            raise ImportError("pandas is required for this functionality")
        efficiency_df = pd.DataFrame(data=efficiency_list)
        ## センチメートル単位でフィギュアサイズを設定します
        width_cm = 4.4
        height_cm = 4.0
        ## cmをinchに変換
        width_in = width_cm / 2.54
        height_in = height_cm / 2.54
        ## 指定されたサイズで新しいフィギュアを作成します
        fig, ax = plt.subplots(figsize=(width_in, height_in))

        cmthermal = GraphMaker.generate_mycmap()
        im = sns.heatmap(
            efficiency_df, square=True, cmap=cmthermal, vmax=100, vmin=85, cbar=False
        )
        cbar = plt.colorbar(im.collections[0], ax=ax, shrink=0.8, pad=0.1)
        cax = cbar.ax
        sns.despine()
        ax.invert_yaxis()
        ax.collections[0].colorbar.minorticks_on()
        cax = ax.collections[0].colorbar.ax
        cax.tick_params(which="minor")
        cax.tick_params(which="major", labelsize=7)
        # ax.collections[0].colorbar.set_label(
        #     "Efficiency (%)", fontsize=7
        # )  # カラーバーのラベル

        # 軸ラベルを設定
        tick_positions = np.arange(0, segments + 1, 25)
        # ラベルを作成
        x_labels = []
        y_labels = []

        # h2とh3の値を計算し、小数点第一位に丸めてリストに追加
        for i in range(len(tick_positions)):
            y_value = i * 10.0
            r1_value = visualize.r1_min + i * 6.0

            # 小数点第一位に丸めて、f-stringで文字列化
            x_labels.append(f"{y_value:.1f}")
            y_labels.append(f"{r1_value:.1f}")
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(x_labels, rotation=45, fontsize=7)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(y_labels, fontsize=7)

        plt.xlabel(r"position($\it{y}$) (cm)", fontsize=7)
        plt.ylabel(r"$\it{r}_1$ (cm)", fontsize=7)
        plt.tight_layout()
        plt.savefig("changeEfficiencyByRoiAndMovement.jpg", format="jpg", dpi=500)
        plt.show()

    @staticmethod
    def search_near_point(aimpoint, points, constraints):
        """Search for nearest point with constraints."""
        neardistance = 1000
        nearestpoint = None
        for search_index in range(len(points)):
            searchpoint = points[search_index]
            if GraphMaker.check_search_point(aimpoint, searchpoint, constraints):
                distance = math.sqrt(
                    (searchpoint[0] - aimpoint[0]) ** 2
                    + (searchpoint[1] - aimpoint[1]) ** 2
                    + (searchpoint[2] - aimpoint[2]) ** 2
                )
                if distance < neardistance:
                    neardistance = distance
                    nearestpoint = search_index
        return nearestpoint

    @staticmethod
    def check_search_point(aimpoint, searchpoint, constraints):
        """Check if search point meets constraints."""
        constraint_x = (searchpoint[0] - aimpoint[0]) * constraints[0]
        constraint_y = (searchpoint[1] - aimpoint[1]) * constraints[1]
        constraint_z = (searchpoint[2] - aimpoint[2]) * constraints[2]
        return bool(constraint_x and constraint_y and constraint_z)

    @staticmethod
    def change_efficiency_by_double_roi() -> None:
        """Change efficiency by double ROI analysis."""
        shield_number = 90
        visualize = Visualize(shield_number)
        visualize.y_move = 13.0
        visualize.z_move = 11.0
        segments = 50

        efficiency_list = np.array([])
        for i in range(segments + 1):
            visualize.h3 = i * 4 / segments
            visualize.update_values_for_shape(
                changed_h1=0,
                changed_h2=0,
                changed_h3=1,
                changed_r1=0,
                changed_r2=0,
                changed_r3=0,
                changed_r4=0,
            )
            for j in range(segments + 1):
                visualize.h2 = j * 4 / segments
                visualize.update_values_for_shape(
                    changed_h1=0,
                    changed_h2=1,
                    changed_h3=0,
                    changed_r1=0,
                    changed_r2=0,
                    changed_r3=0,
                    changed_r4=0,
                )
                efficiency = visualize.get_efficiency() * 100
                efficiency_list = np.append(efficiency_list, efficiency)

        efficiency_list = efficiency_list.reshape((segments + 1, segments + 1))
        print(efficiency_list)
        if pd is None:
            raise ImportError("pandas is required for this functionality")
        efficiency_df = pd.DataFrame(data=efficiency_list)
        ## センチメートル単位でフィギュアサイズを設定します
        width_cm = 5.5
        height_cm = 4.5
        ## cmをinchに変換
        width_in = width_cm / 2.54
        height_in = height_cm / 2.54
        ## 指定されたサイズで新しいフィギュアを作成します
        fig, ax = plt.subplots(figsize=(width_in, height_in))

        cmthermal = GraphMaker.generate_mycmap()
        sns.heatmap(efficiency_df, square=True, cmap=cmthermal, vmax=98, vmin=95)
        sns.despine()
        ax.invert_yaxis()
        ax.collections[0].colorbar.minorticks_on()
        cax = ax.collections[0].colorbar.ax
        cax.tick_params(which="minor")
        cax.tick_params(which="major", labelsize=7)
        ax.collections[0].colorbar.set_label(
            "Efficiency (%)", fontsize=7
        )  # カラーバーのラベル

        # 軸ラベルを設定
        tick_positions = np.arange(0, segments + 1, 25)
        # ラベルを作成
        x_labels = []
        y_labels = []

        # h2とh3の値を計算し、小数点第一位に丸めてリストに追加
        for i in range(len(tick_positions)):
            y_value = i * 0.8
            r1_value = i * 0.8

            # 小数点第一位に丸めて、f-stringで文字列化
            x_labels.append(f"{y_value:.1f}")
            y_labels.append(f"{r1_value:.1f}")
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(x_labels, rotation=45, fontsize=7)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(y_labels, fontsize=7)

        plt.xlabel(r"$\it{h}_2$ (cm)", fontsize=7)
        plt.ylabel(r"$\it{h}_3$ (cm)", fontsize=7)
        plt.tight_layout()
        plt.savefig("changeEfficiencyByDoubleRoi.jpg", format="jpg", dpi=500)
        plt.show()

    @staticmethod
    def generate_mycmap():
        """Generate custom colormap for visualizations."""
        colors = ["#1c3f75", "#068fb9", "#f1e235", "#d64e8b", "#730e22"]
        cmap_name = "cmthermal"
        values = range(len(colors))
        vmax = np.ceil(np.max(values))
        color_list = []
        for vi, ci in zip(values, colors, strict=False):
            color_list.append((vi / vmax, ci))

        return mcolors.LinearSegmentedColormap.from_list(cmap_name, color_list, 256)
