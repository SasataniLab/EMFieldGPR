"""File for making Aligned Edge Polycube mesh."""

from pathlib import Path

import numpy as np
from stl import mesh

from EMFieldML.config import config, get_logger, paths, template
from EMFieldML.Modeling.BVH import BVH
from EMFieldML.Modeling.make_first_grid import (
    make_first_out_points,
    make_first_point_edge_plane,
)
from EMFieldML.Modeling.make_second_grid import make_second_grid, make_second_out_points

logger = get_logger(__name__)


class PolyCubeMaker:
    """Class for creating and managing PolyCube mesh models.
    Shield Geometry and Aligned-Edge Polycube Mesh
    This process generates a computational mesh for the shield geometry
    using an aligned-edge polycube approach.

    Initial Point Generation: The mesh begins with a set of initial points that include
    both corner and surface points, forming cubes that precisely fit the shield.
    Mesh Subdivision: The initial mesh is then subdivided by interpolating new points
    between the existing ones. By aligning the polycube edges to the geometry,
    this method minimizes the number of points required to accurately represent the shield.

    Adaptive Exterior Grid Generation
    An adaptive exterior grid is generated to support electromagnetic field prediction.

    Initial Positioning: Exterior cubes are initially positioned around the shield-fitting polycubes.
    They are adjusted to conform to the shield geometry's shape.
    Subdivision: These exterior polycubes are further subdivided through interpolation relative to
    the exterior cubes, creating a refined grid.
    Dynamic Adjustment: This adaptive exterior grid is designed to maintain its relative position to
    the transmitter and receiver coils. This adjustment accounts for any changes in the coils' relative
    positions and helps prevent abrupt variations in the magnetic field predictions that would otherwise
    occur with movement.

    For further details, please refer to Figure 2 in the paper.
    """

    @staticmethod
    def make_modeling(
        input_path: Path,
        output_path_polycube: Path,
        output_path_exterior: Path,
    ) -> None:
        """Create PolyCube mesh modeling from STL file.
        Aligned-Edge Polycube Mesh Generation
        The Aligned-Edge Polycube Mesh is created by fitting large cubes to
        the geometry's shape and then subdividing them into a finer mesh.

        Adaptive Exterior Grid Generation
        The Adaptive Exterior Grid is created by moving points on a base grid to
        align with the Aligned-Edge Polycube Mesh, followed by further subdivision.

        This process is refined up to a maximum of three levels.

        Args:
            input_path: Path to the STL file
            output_path_polycube: Path to save the PolyCube mesh file
            output_path_exterior: Path to save the exterior grid file

        Returns:
            None

        """
        # Perform BVH calculation for 3D data file. This only needs to be done once initially
        nodes = BVH(input_path)

        # Convert STL file to processable data format
        your_mesh = mesh.Mesh.from_file(input_path)
        model_points = np.unique(your_mesh.points.reshape([-1, 3]), axis=0)

        # Calculate initial configuration for TX side points (level 1)
        points, edges, planes = make_first_point_edge_plane(nodes, model_points)

        # Calculate initial configuration for prediction point positions (level 1)
        out_points, boxes, edges_outside, planes_outside = make_first_out_points(
            points, planes
        )

        # Calculate next configuration for TX side points (level 2)
        points, edges, planes = make_second_grid(nodes, points, edges, planes)

        # Calculate next configuration for prediction point positions (level 2)
        out_points, boxes, edges_outside, planes_outside = make_second_out_points(
            out_points, boxes, edges_outside, planes_outside, points, edges, planes
        )

        # Calculate next configuration for TX side points (level 3)
        points, edges, planes = make_second_grid(nodes, points, edges, planes)

        # Calculate next configuration for prediction point positions (level 3)
        out_points, boxes, edges_outside, planes_outside = make_second_out_points(
            out_points, boxes, edges_outside, planes_outside, points, edges, planes
        )

        ##### Save PolyCube mesh #####
        # Array to save TX point positions
        points_for_3D_TX = np.array([points[i].xyz for i in range(len(points))])

        # Save if file path is specified
        if output_path_polycube is not None:
            with Path(output_path_polycube).open("w") as f:
                for i in range(len(points_for_3D_TX)):
                    for j in range(3):
                        print(points[i].xyz[j], file=f)
            logger.info(f"Saved {output_path_polycube}")

        ##### Exterior Grid の保存 #####
        # 予測の点の位置を保存するための配列
        points_for_3D_prediction = np.empty((0, 3), float)

        for i in range(len(out_points)):
            # TXやRXの表面にあるものは除外
            if out_points[i].mode == 0:
                points_for_3D_prediction = np.append(
                    points_for_3D_prediction, [out_points[i].xyz], axis=0
                )

        # Save if file path is specified
        if output_path_exterior is not None:
            with Path(output_path_exterior).open("w") as f:
                for i in range(len(points_for_3D_prediction)):
                    print(
                        points_for_3D_prediction[i][0],
                        points_for_3D_prediction[i][1],
                        points_for_3D_prediction[i][2],
                        file=f,
                    )
            logger.info(f"Saved {output_path_exterior}")

    @staticmethod
    def make_modeling_all(
        n_shield_shape: int = config.n_shield_shape,
        path_stl_dir: Path = paths.CIRCLE_STL_DIR,
        path_stl_file: str = template.circle_stl,
        path_polycube_dir: Path = paths.CIRCLE_X_DIR,
        path_polycube_file: str = template.circle_x,
        path_exterior_dir: Path = paths.CIRCLE_PREDICTION_POINT_DIR,
        path_exterior_file: str = template.circle_prediction_point,
    ) -> None:
        """Create PolyCube mesh modeling for all shield shapes.
        This function processes multiple STL files to generate corresponding
        PolyCube mesh models and exterior grids.

        Args:
            n_shield_shape: Number of shield shapes to process
            path_stl_dir: Directory containing STL files
            path_stl_file: STL file names
            path_polycube_dir: Directory to save PolyCube mesh files
            path_polycube_file: PolyCube mesh file names
            path_exterior_dir: Directory to save exterior grid files
            path_exterior_file: Exterior grid file names

        Returns:
            None

        """
        for shield_index in range(1, n_shield_shape + 1):
            input_path = path_stl_dir / path_stl_file.format(index=shield_index)
            output_path_polycube = path_polycube_dir / path_polycube_file.format(
                index=shield_index
            )
            output_path_exterior = path_exterior_dir / path_exterior_file.format(
                index=shield_index
            )
            PolyCubeMaker.make_modeling(
                input_path=input_path,
                output_path_polycube=output_path_polycube,
                output_path_exterior=output_path_exterior,
            )

    @staticmethod
    def move_shield_polycube(
        input_path: Path,
        shield_index: int,
        path_output_dir: Path,
        path_output_file: str,
        n_shield_shape: int = config.n_shield_shape,
        y_grid: int = config.y_grid,
        z_grid: int = config.z_grid,
        y_step: float = config.y_step,
        z_step: float = config.z_step,
    ) -> None:
        """Move shield polycube with grid-based positioning.

        Input Data Generation for Machine Learning
        The input data for the machine learning model is generated from the polycube mesh and the coil positions.
        Mesh Point Coordinates: This includes the XYZ coordinates of each point on the polycube mesh.
        Coil Coordinates: This includes the YZ coordinates of the moved (transformed) coils.

        Args:
            input_path: Path to the PolyCube mesh file
            shield_index: Index of the shield shape
            path_output_dir: Directory to save output files
            path_output_file: Template for output file names
            n_shield_shape: Total number of shield shapes
            y_grid: Number of grid points of coil move in the Y direction
            z_grid: Number of grid points of coil move in the Z direction
            y_step: Step size of coil move in the Y direction
            z_step: Step size of coil move in the Z direction

        Returns:
            None

        """
        with Path(input_path).open() as f:
            l_strip = [float(s.rstrip()) for s in f.readlines()]

        for y in range(y_grid):
            for z in range(z_grid):
                y_move = y * y_step
                z_move = z * z_step
                output_path = path_output_dir / path_output_file.format(
                    index=shield_index
                    + y * n_shield_shape * z_grid
                    + z * n_shield_shape
                )
                # Need to check this export filename
                with Path(output_path).open("w") as f:
                    for j in range(len(l_strip)):
                        print(l_strip[j], file=f)
                    print(y_move, file=f)
                    print(z_move, file=f)
                logger.info(f"Saved {output_path}")

    @staticmethod
    def move_shield_exterior(
        input_path_polycube: Path,
        input_path_exterior: Path,
        number: int,
        y: float,
        z: float,
        output_path_dir: Path = paths.CIRCLE_PREDICTION_POINT_DIR,
        output_path_file: str = template.circle_prediction_point_move,
        n_prediction_points: int = config.n_prediction_points_level3,
    ) -> None:
        """Move shield exterior points with linear blend skinning.
        Linear Blend Skinning (LBS)
        This module provides an implementation of Linear Blend Skinning (LBS), a computationally efficient algorithm
        for deforming a 3D mesh based on a skeletal structure.
        LBS is the industry standard for character animation in computer graphics, including video games and films.

        Core Principle
        LBS operates on the principle that the deformation of a mesh's vertices is controlled by an underlying
        skeleton of "bones" or joints. A key challenge, such as deforming the mesh around an elbow joint,
        is to handle vertices that are influenced by multiple bones (e.g., both the upper arm and forearm bones).

        This is solved by assigning a "weight" to each vertex for every bone that influences it.
        These weights define the degree of influence each bone has on the vertex's final position.
        The sum of all weights for a single vertex must equal 1.0.

        For further details, please refer to equation (6) in the paper.

        Args:
            input_path_polycube: Path to the PolyCube mesh file
            input_path_exterior: Path to the exterior grid file
            number: Unique identifier for the current configuration
            y: Y-coordinate of the coil position
            z: Z-coordinate of the coil position
            output_path_dir: Directory to save output files
            output_path_file: Output file names
            n_prediction_points: Number of prediction points to process

        Returns:
            None

        """
        # 移動する前の予測点
        prediction_point = np.loadtxt(input_path_exterior)

        # フェライトシールドの点
        with Path(input_path_polycube).open() as f:
            l_strip = [float(s.rstrip()) for s in f.readlines()]

        # TX(送信器)、RX(受信器)のフェライトシールドの点。それぞれ72点なのは基準点で、これらのみを元にlinearblendskinningを行うため
        TX_ferrite_point = np.array(
            [[0.0, 0.0, 0.0] for _ in range(config.n_initial_points)]
        )
        RX_ferrite_point = np.array(
            [[0.0, 0.0, 0.0] for _ in range(config.n_initial_points)]
        )
        for i in range(config.n_initial_points):
            TX_ferrite_point[i][0] = l_strip[i * 3]
            TX_ferrite_point[i][1] = l_strip[i * 3 + 1]
            TX_ferrite_point[i][2] = l_strip[i * 3 + 2]
            RX_ferrite_point[i][0] = l_strip[i * 3]
            RX_ferrite_point[i][1] = l_strip[i * 3 + 1]
            RX_ferrite_point[i][2] = config.initial_coil_distance - l_strip[i * 3 + 2]

        # 予測点それぞれに対してすべての基準点の重みを格納
        weight = np.zeros((n_prediction_points, config.n_initial_points * 2))
        for i in range(n_prediction_points):
            total_weight = 0  # @HonjoYuichi: Renamed from 'sum' to 'total_weight' to avoid shadowing builtin
            # それぞれのフェライトの点に対して重みを計算して合計する
            for j in range(config.n_initial_points * 2):
                if j < config.n_initial_points:
                    d = np.linalg.norm(TX_ferrite_point[j] - prediction_point[i], ord=2)
                    total_weight += 1 / d ** (config.LBS_w_move)
                else:
                    d = np.linalg.norm(
                        RX_ferrite_point[j - config.n_initial_points]
                        - prediction_point[i],
                        ord=2,
                    )
                    total_weight += 1 / d ** (config.LBS_w_move)

            # それぞれのフェライトの点に対する重みを正規化して、本当の重みとして保存する
            for j in range(config.n_initial_points * 2):
                if j < config.n_initial_points:
                    d = np.linalg.norm(TX_ferrite_point[j] - prediction_point[i], ord=2)
                    weight[i][j] = 1 / d ** (config.LBS_w_move) / sum
                else:
                    d = np.linalg.norm(
                        RX_ferrite_point[j - config.n_initial_points]
                        - prediction_point[i],
                        ord=2,
                    )
                    weight[i][j] = 1 / d ** (config.LBS_w_move) / sum

        # フェライトシールドの各点の変化。RXのみy,zが動く
        change = np.array([[0.0, 0.0, 0.0] for _ in range(config.n_initial_points * 2)])
        for i in range(config.n_initial_points, config.n_initial_points * 2):
            change[i][1] = y
            change[i][2] = z

        # prediction_pointにフェライトシールドの変化分を足す
        for i in range(n_prediction_points):
            for j in range(config.n_initial_points * 2):
                prediction_point[i] += change[j] * weight[i][j]

        # 最後に新しいprediction_pointを保存する
        output_path = output_path_dir / output_path_file.format(index=number)
        with Path(output_path).open("w") as f:
            for i in range(n_prediction_points):
                print(
                    prediction_point[i][0],
                    prediction_point[i][1],
                    prediction_point[i][2],
                    file=f,
                )

        logger.info(f"Saved {output_path}")

    @staticmethod
    def move_shield_polycube_all(
        n_shield_shape: int = config.n_shield_shape,
        path_polycube_dir: Path = paths.CIRCLE_X_DIR,
        path_polycube_file: str = template.circle_x,
        path_output_dir: Path = paths.CIRCLE_MOVE_X_DIR,
        path_output_file: str = template.circle_move_x,
    ) -> None:
        """
        Add position data to the ferrite shape data created so far.
        This creates 169 data for one shape.

        Args:
            n_shield_shape: Number of shield shapes to process
            path_polycube_dir: Directory containing polycube data files
            path_polycube_file: Template for polycube file names
            path_output_dir: Directory to save output files
            path_output_file: Template for output file names

        """
        for shield_index in range(1, n_shield_shape + 1):
            input_path_polycube = path_polycube_dir / path_polycube_file.format(
                index=shield_index
            )
            PolyCubeMaker.move_shield_polycube(
                input_path=input_path_polycube,
                shield_index=shield_index,
                path_output_dir=path_output_dir,
                path_output_file=path_output_file,
            )

    @staticmethod
    def move_shield_exterior_need(
        n_shield_shape: int = None,
        input_path: Path = None,
    ) -> None:
        """
        Make Exterior Grid for selected shields and positions.
        """
        # Set defaults if not provided
        if n_shield_shape is None:
            n_shield_shape = config.n_shield_shape
        if input_path is None:
            input_path = paths.TRAIN_DIR / template.x_train_init

        with Path(input_path).open() as f:
            train_list = [int(s.rstrip()) for s in f.readlines()]

        for number in train_list:
            shield_index = number % n_shield_shape
            if shield_index == 0:
                shield_index = n_shield_shape
            input_path_polycube = paths.CIRCLE_X_DIR / template.circle_x.format(
                index=shield_index
            )
            input_path_exterior = (
                paths.CIRCLE_PREDICTION_POINT_DIR
                / template.circle_prediction_point.format(index=shield_index)
            )

            PolyCubeMaker.move_shield_exterior(
                input_path_polycube=input_path_polycube,
                input_path_exterior=input_path_exterior,
                number=number,
                y=(number - shield_index) // n_shield_shape // config.z_grid,
                z=(number - shield_index) // n_shield_shape % config.z_grid,
            )
