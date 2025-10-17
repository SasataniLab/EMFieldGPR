"""Module for creating Y data for machine learning models."""

import bisect
import csv
import math
from pathlib import Path

import numpy as np

from EMFieldML.config import config, get_logger, paths, template

logger = get_logger(__name__)


class TargetDataBuilder:
    """Class for creating Y data for machine learning models.

    The ground truth data is created for machine learning.
    The ground truth data here consists of the magnetic field magnitude,
    vector direction, and efficiency at each point.
    """

    @staticmethod
    def bisect(
        point_data_list: list[int],
        point: float,
    ) -> int:
        """
        Search for the position of the value within the list.

        Args:
            point_data_list: List of data points to search through
            point: The value to search for in the vicinity

        Returns:
            int: Index of the closest lower value in the list

        """  # noqa: D409
        # Find the insertion point for point in point_data to maintain sorted order
        index = bisect.bisect(point_data_list, point)

        # Adjust the index to return the appropriate value
        if index == len(point_data_list):
            return len(point_data_list) - 2
        return index - 1

    @staticmethod
    def calculate_magnitude(
        Hx: float,
        Hx_theta: float,
        Hy: float,
        Hy_theta: float,
        Hz: float,
        Hz_theta: float,
    ) -> float:
        """Calculate magnitude from field components.

        The magnetic field calculated by the FEKO simulator consists of
        a magnitude and a phase for each of the x, y, and z components.
        The magnetic field is therefore represented by an equation similar
        to the one below:

        MagneticField = Hx * cos(theta + Hx_theta) + Hy * cos(theta + Hy_theta) + Hz * cos(theta + Hz_theta)

        From this, the maximum value of the magnetic field is calculated
        over the theta.

        Args:
            Hx: Magnitude of the x-component of the magnetic field
            Hx_theta: Phase of the x-component of the magnetic field (degrees)
            Hy: Magnitude of the y-component of the magnetic field
            Hy_theta: Phase of the y-component of the magnetic field (degrees)
            Hz: Magnitude of the z-component of the magnetic field
            Hz_theta: Phase of the z-component of the magnetic field (degrees)

        Returns:
            float: Maximum magnitude of the magnetic field

        """
        A = (
            Hx**2 * math.sin(math.radians(2 * Hx_theta))
            + Hy**2 * math.sin(math.radians(2 * Hy_theta))
            + Hz**2 * math.sin(math.radians(2 * Hz_theta))
        )
        B = (
            Hx**2 * math.cos(math.radians(2 * Hx_theta))
            + Hy**2 * math.cos(math.radians(2 * Hy_theta))
            + Hz**2 * math.cos(math.radians(2 * Hz_theta))
        )
        theta_1 = (-math.degrees(math.acos(B / (A**2 + B**2) ** (1 / 2)))) / 2.0
        theta_2 = theta_1 * -1
        theta_3 = (180 - math.degrees(math.acos(B / (A**2 + B**2) ** (1 / 2)))) / 2.0
        theta_4 = (180 + math.degrees(math.acos(B / (A**2 + B**2) ** (1 / 2)))) / 2.0
        ans = 0
        ans_2 = (
            Hx**2 * math.cos(math.radians(theta_1 + Hx_theta)) ** 2
            + Hy**2 * math.cos(math.radians(theta_1 + Hy_theta)) ** 2
            + Hz**2 * math.cos(math.radians(theta_1 + Hz_theta)) ** 2
        )
        ans_3 = (
            Hx**2 * math.cos(math.radians(theta_2 + Hx_theta)) ** 2
            + Hy**2 * math.cos(math.radians(theta_2 + Hy_theta)) ** 2
            + Hz**2 * math.cos(math.radians(theta_2 + Hz_theta)) ** 2
        )
        ans_4 = (
            Hx**2 * math.cos(math.radians(theta_3 + Hx_theta)) ** 2
            + Hy**2 * math.cos(math.radians(theta_3 + Hy_theta)) ** 2
            + Hz**2 * math.cos(math.radians(theta_3 + Hz_theta)) ** 2
        )
        ans_5 = (
            Hx**2 * math.cos(math.radians(theta_4 + Hx_theta)) ** 2
            + Hy**2 * math.cos(math.radians(theta_4 + Hy_theta)) ** 2
            + Hz**2 * math.cos(math.radians(theta_4 + Hz_theta)) ** 2
        )
        ans = max(ans, ans_2)
        ans = max(ans, ans_3)
        ans = max(ans, ans_4)
        ans = max(ans, ans_5)
        return ans ** (1 / 2)

    @staticmethod
    def write_data(
        data: list[list[float]],
        input_path_raw_data: Path,
        input_path_prediction_point: Path,
        prediction_point_list: list[int],
    ) -> list[list[float]]:
        """Write data to files with prediction point processing.

        FEKO can calculate the magnetic field only at fixed points.
        To apply this to points between those fixed points, it's possible
        to use trilinear interpolation, a 3D linear interpolation method.
        It's based on a linear interpolation of the distances from each vertex of a cube.

        Args:
            data: List to store the calculated data
            input_path_raw_data: Path to the FEKO output file
            input_path_prediction_point: Path to the file containing prediction points
            prediction_point_list: List of indices for prediction points to process

        Returns:
            list[list[float]]: Updated data list with calculated values

        """
        prediction_point = np.loadtxt(input_path_prediction_point)

        with Path(input_path_raw_data).open() as f:
            lines = f.readlines()

        lines_strip = [line.strip() for line in lines]
        list_rownum = [
            i for i, line_s in enumerate(lines_strip) if "RESULTS FOR LOADS" in line_s
        ]
        line_split_efficiency = lines_strip[list_rownum[0] + 6].split()
        rx_power = float(line_split_efficiency[10])

        lines_strip = [line.strip() for line in lines]
        list_rownum = [
            i
            for i, line_s in enumerate(lines_strip)
            if "VALUES OF THE MAGNETIC FIELD STRENGTH in A/m" in line_s
        ]
        x_point_list = config.y_data_x_point_list
        y_point_list = config.y_data_y_point_list
        z_point_list = config.y_data_z_point_list

        for i in prediction_point_list:
            x_index = TargetDataBuilder.bisect(x_point_list, prediction_point[i][0])
            y_index = TargetDataBuilder.bisect(y_point_list, prediction_point[i][1])
            z_index = TargetDataBuilder.bisect(z_point_list, prediction_point[i][2])
            ans = 0
            basedPointA = [
                x_point_list[x_index],
                y_point_list[y_index],
                z_point_list[z_index],
            ]
            basedPointB = [
                x_point_list[x_index + 1],
                y_point_list[y_index + 1],
                z_point_list[z_index + 1],
            ]
            for xx in range(x_index, x_index + 2):
                for yy in range(y_index, y_index + 2):
                    for zz in range(z_index, z_index + 2):
                        line_split = lines_strip[
                            list_rownum[0]
                            + 6
                            + xx
                            + yy * config.y_data_x_point_num
                            + zz * config.y_data_x_point_num * config.y_data_y_point_num
                        ].split()
                        magnetic_field = TargetDataBuilder.calculate_magnitude(
                            float(line_split[4]),
                            float(line_split[5]),
                            float(line_split[6]),
                            float(line_split[7]),
                            float(line_split[8]),
                            float(line_split[9]),
                        )
                        nowPoint = [
                            x_point_list[xx],
                            y_point_list[yy],
                            z_point_list[zz],
                        ]
                        ans += (
                            TargetDataBuilder.calculateTrilinearWeight(
                                prediction_point[i], nowPoint, basedPointA, basedPointB
                            )
                            * magnetic_field
                        )
                        # Alternative trilinear interpolation calculation (commented out)
                        # ans += abs(1 - abs(pre_x - x_point_list[xx]) / abs(x_point_list[x_index+1] - x_point_list[x_index])) \
                        #         * abs(1 - abs(pre_y - y_point_list[yy]) / abs(y_point_list[y_index+1] - y_point_list[y_index])) \
                        #         * abs(1 - abs(pre_z - z_point_list[zz]) / abs(z_point_list[z_index+1] - z_point_list[z_index])) \
                        #         * magnetic_field
            data[i].append([ans / (rx_power ** (1 / 2))])

        return data

    @staticmethod
    def make_new_ydatafile(
        x_train_list: list[int],
        output_dir: Path,
        input_path_raw_data_dir: Path = paths.CIRCLE_FEKO_RAW_DIR,
        input_path_prediction_point_dir: Path = paths.CIRCLE_PREDICTION_POINT_DIR,
        input_path_prediction_point_file: str = template.circle_prediction_point_move,
        output_file: str = template.y_data_magnitude,
        prediction_point_list: list = None,
        n_prediction_points: int = None,
    ) -> None:
        """Create new Y data file for training.

        The data generation method employs a linear interpolation
        of the distances from each vertex of a cube.

        Args:
            x_train_list: List of training data indices
            output_dir: Directory to save the output file
            input_path_raw_data_dir: Directory containing raw FEKO data files
            input_path_prediction_point_dir: Directory containing prediction point files
            input_path_prediction_point_file: File name containing prediction point positions
            output_file: output file names
            prediction_point_list: List of indices for prediction points to process
            n_prediction_points: Total number of prediction points

        Returns:
            None

        """
        # Set defaults if not provided
        if prediction_point_list is None:
            prediction_point_list = list(range(config.n_prediction_points_level3))
        if n_prediction_points is None:
            n_prediction_points = config.n_prediction_points_level3

        data = [[] for _ in range(n_prediction_points)]

        for index in x_train_list:
            input_path_raw_data = input_path_raw_data_dir / template.raw_data.format(
                index=index
            )
            input_path_prediction_point = (
                input_path_prediction_point_dir
                / input_path_prediction_point_file.format(index=index)
            )
            data = TargetDataBuilder.write_data(
                data,
                input_path_raw_data,
                input_path_prediction_point,
                prediction_point_list,
            )

        for index in prediction_point_list:
            output_path = output_dir / output_file.format(index=index)
            with Path(output_path).open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(data[index])

            logger.info(f"Saved {output_path}")

    @staticmethod
    def calculate_vector(Hx, Hx_theta, Hy, Hy_theta, Hz, Hz_theta):
        """Calculate vector components from field data.

        The magnetic field calculated by the FEKO simulator consists of
        a magnitude and a phase for each of the x, y, and z components.
        The magnetic field is therefore represented by an equation similar
        to the one below:

        MagneticField = Hx * cos(theta + Hx_theta) + Hy * cos(theta + Hy_theta) + Hz * cos(theta + Hz_theta)

        The vector's direction is calculated based on this equation.
        However, theta can be any arbitrary value, and in this case,
        the theta = -140 has been selected as the theta at which
        the vector at the center of the coil becomes maximal.

        The vector is converted into a polar coordinate system.

        Args:
            Hx: Magnitude of the x-component of the magnetic field
            Hx_theta: Phase of the x-component of the magnetic field (degrees)
            Hy: Magnitude of the y-component of the magnetic field
            Hy_theta: Phase of the y-component of the magnetic field (degrees)
            Hz: Magnitude of the z-component of the magnetic field
            Hz_theta: Phase of the z-component of the magnetic field (degrees)

        Returns:
            tuple: (theta, phi) angles in radians representing the vector direction

        """
        theta = -140
        x = Hx * math.cos(math.radians(theta + Hx_theta))
        y = Hy * math.cos(math.radians(theta + Hy_theta))
        z = Hz * math.cos(math.radians(theta + Hz_theta))
        if x >= 0:
            ans_theta = math.atan(y / x)
        else:
            ans_theta = math.atan(y / x)
            if ans_theta >= 0:
                ans_theta -= math.pi
            else:
                ans_theta += math.pi
        ans_phi = math.asin(z / (x**2 + y**2 + z**2) ** (1 / 2))
        return ans_theta, ans_phi

    @staticmethod
    def write_data_vector(
        data_theta: list[list[float]],
        data_phi: list[list[float]],
        input_path_raw_data: Path,
        input_path_prediction_point: Path,
        prediction_point_list: list[int],
    ):
        """Write vector data to files.
        Three-dimensional linear interpolation from each vertex
        of a cube is also applied to the vectors.

        Args:
            data_theta: List to store theta components of the vector
            data_phi: List to store phi components of the vector
            input_path_raw_data: Path to the FEKO output file
            input_path_prediction_point: Path to the file containing prediction point positions
            prediction_point_list: List of indices for prediction points to process

        Returns:
            tuple: Updated data_theta and data_phi lists with calculated values

        """
        prediction_point = np.loadtxt(input_path_prediction_point)

        with Path(input_path_raw_data).open() as f:
            lines = f.readlines()

        lines_strip = [line.strip() for line in lines]
        list_rownum = [
            i
            for i, line_s in enumerate(lines_strip)
            if "VALUES OF THE MAGNETIC FIELD STRENGTH in A/m" in line_s
        ]
        x_point_list = config.y_data_x_point_list
        y_point_list = config.y_data_y_point_list
        z_point_list = config.y_data_z_point_list

        for i in prediction_point_list:
            x_index = TargetDataBuilder.bisect(x_point_list, prediction_point[i][0])
            y_index = TargetDataBuilder.bisect(y_point_list, prediction_point[i][1])
            z_index = TargetDataBuilder.bisect(z_point_list, prediction_point[i][2])
            ans_theta = 0
            ans_phi = 0
            basedPointA = [
                x_point_list[x_index],
                y_point_list[y_index],
                z_point_list[z_index],
            ]
            basedPointB = [
                x_point_list[x_index + 1],
                y_point_list[y_index + 1],
                z_point_list[z_index + 1],
            ]
            for xx in range(x_index, x_index + 2):
                for yy in range(y_index, y_index + 2):
                    for zz in range(z_index, z_index + 2):
                        line_split = lines_strip[
                            list_rownum[0]
                            + 6
                            + xx
                            + yy * config.y_data_x_point_num
                            + zz * config.y_data_x_point_num * config.y_data_y_point_num
                        ].split()
                        theta, phi = TargetDataBuilder.calculate_vector(
                            float(line_split[4]),
                            float(line_split[5]),
                            float(line_split[6]),
                            float(line_split[7]),
                            float(line_split[8]),
                            float(line_split[9]),
                        )
                        nowPoint = [
                            x_point_list[xx],
                            y_point_list[yy],
                            z_point_list[zz],
                        ]
                        ans_theta += (
                            TargetDataBuilder.calculateTrilinearWeight(
                                prediction_point[i], nowPoint, basedPointA, basedPointB
                            )
                            * theta
                        )
                        ans_phi += (
                            TargetDataBuilder.calculateTrilinearWeight(
                                prediction_point[i], nowPoint, basedPointA, basedPointB
                            )
                            * phi
                        )
            data_theta[i].append([theta])
            data_phi[i].append([phi])

        return data_theta, data_phi

    @staticmethod
    def calculateTrilinearWeight(
        prePoint: list[float],
        nowPoint: list[float],
        basedPointA: list[float],
        basedPointB: list[float],
    ) -> float:
        """
        Calculate the trilinear weight for the given point.
        """
        return (
            (1 - abs(prePoint[0] - nowPoint[0]) / abs(basedPointB[0] - basedPointA[0]))
            * (
                1
                - abs(prePoint[1] - nowPoint[1]) / abs(basedPointB[1] - basedPointA[1])
            )
            * (
                1
                - abs(prePoint[2] - nowPoint[2]) / abs(basedPointB[2] - basedPointA[2])
            )
        )

    @staticmethod
    def make_new_ydatafile_vector(
        x_train_list: list[int],
        output_dir: Path,
        input_path_raw_data_dir=paths.CIRCLE_FEKO_RAW_DIR,
        input_path_prediction_point_dir=paths.CIRCLE_PREDICTION_POINT_DIR,
        input_path_prediction_point_file=template.circle_prediction_point_move,
        output_file_theta: str = template.y_data_vector_theta,
        output_file_phi: str = template.y_data_vector_phi,
        prediction_point_list=None,
        n_prediction_points: int = None,
    ):
        """Create new Y data file for vector training.
        Three-dimensional linear interpolation from each vertex
        of a cube is also applied to the vectors.

        Args:
            x_train_list: List of training data indices
            output_dir: Directory to save the output file
            input_path_raw_data_dir: Directory containing raw FEKO data files
            input_path_prediction_point_dir: Directory containing prediction point files
            input_path_prediction_point_file: File name containing prediction point positions
            output_file_theta: output file names for theta component
            output_file_phi: output file names for phi component
            prediction_point_list: List of indices for prediction points to process
            n_prediction_points: Total number of prediction points

        Returns:
            None

        """
        # Set defaults if not provided
        if prediction_point_list is None:
            prediction_point_list = list(range(config.n_prediction_points_level3))
        if n_prediction_points is None:
            n_prediction_points = config.n_prediction_points_level3

        data_theta = [[] for _ in range(n_prediction_points)]
        data_phi = [[] for _ in range(n_prediction_points)]

        for index in x_train_list:
            input_path_raw_data = input_path_raw_data_dir / template.raw_data.format(
                index=index
            )
            input_path_prediction_point = (
                input_path_prediction_point_dir
                / input_path_prediction_point_file.format(index=index)
            )
            TargetDataBuilder.write_data_vector(
                data_theta,
                data_phi,
                input_path_raw_data,
                input_path_prediction_point,
                prediction_point_list,
            )

        for index in prediction_point_list:
            output_path = output_dir / output_file_theta.format(index=index)
            with Path(output_path).open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(data_theta[index])

            logger.info(f"Saved {output_path}")

        for index in prediction_point_list:
            output_path = output_dir / output_file_phi.format(index=index)
            with Path(output_path).open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(data_phi[index])

            logger.info(f"Saved {output_path}")

    @staticmethod
    def write_data_efficiency(
        data: list[list[float]],
        input_path_raw_data: Path,
    ) -> list[float]:
        """
        Calculate the efficiency from the received out file and store it in the data.
        """
        with Path(input_path_raw_data).open() as f:
            lines = f.readlines()

        lines_strip = [line.strip() for line in lines]
        list_rownum = [
            i for i, line_s in enumerate(lines_strip) if "Power in Watt" in line_s
        ]
        line_split_efficiency = lines_strip[list_rownum[0]].split()
        tx_power = float(line_split_efficiency[3])

        lines_strip = [line.strip() for line in lines]
        list_rownum = [
            i for i, line_s in enumerate(lines_strip) if "RESULTS FOR LOADS" in line_s
        ]
        line_split_efficiency = lines_strip[list_rownum[0] + 6].split()
        rx_power = float(line_split_efficiency[10])

        real_efficiency = rx_power / tx_power

        data.append([real_efficiency])

        return data

    @staticmethod
    def make_new_ydatafile_efficiency(
        x_train_list: list[int],
        output_dir: Path,
        input_path_raw_data_dir=paths.CIRCLE_FEKO_RAW_DIR,
    ):
        """
        Execute the write_data_efficiency function for the out file corresponding to the given X_train, store the results in data, and collectively output them to a file.

        Args:
            x_train_list: List of training data indices
            output_dir: Directory to save the output file
            input_path_raw_data_dir: Directory containing raw FEKO data files

        Returns:
            None

        """
        data = []

        for index in x_train_list:
            input_path_raw_data = input_path_raw_data_dir / template.raw_data.format(
                index=index
            )
            data = TargetDataBuilder.write_data_efficiency(data, input_path_raw_data)

        output_path = output_dir / template.y_data_efficiency
        with Path(output_path).open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(data)

        logger.info(f"Saved {output_path}")
