"""3D electromagnetic field visualization system for ferrite shield design optimization.

This module provides interactive 3D visualization of electromagnetic field predictions
using machine learning models. It handles visualization of ferrite shields, coils,
prediction points, and real-time updates based on geometric parameters.

Key components:
- VisualizationConstants: Configuration constants for the visualization system
- VisualizationUI: User interface controls and interactions
- Visualize: Main visualization class with ML-based field predictions

The system uses Polyscope for 3D rendering and supports real-time updates of:
- Ferrite shield geometry and positioning
- Electromagnetic field magnitude and vector visualization
- Efficiency calculations and predictions
- Interactive parameter adjustment through UI controls
"""

import bisect
import copy
import sys
import threading
import time
from pathlib import Path

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from stl import mesh

from EMFieldML.config import config, get_logger, paths, template
from EMFieldML.Visualize import Deformation, Prediction
from EMFieldML.Visualize.make_first_grid import (
    make_first_out_points,
    make_first_point_edge_plane,
)
from EMFieldML.Visualize.make_second_grid import (
    make_second_grid,
    make_second_out_points,
)

logger = get_logger(__name__)


class VisualizationConstants:
    """Constants used throughout the visualization system."""

    # Point size limits
    POINT_SIZE_MIN = 0.0
    POINT_SIZE_MAX = 10.0
    DEFAULT_POINT_SIZE = 2.0

    # Position limits (cm)
    POSITION_MIN = 0.0
    POSITION_MAX = 24.0

    # Ferrite shield parameter limits
    H1_MIN = 0.5
    H1_MAX = 4.0
    H2_MIN = 0.0
    H2_MAX = 4.0
    H3_MIN = 0.0
    H3_MAX = 4.0
    R1_MIN = 12.0
    R1_MAX = 24.0
    R2_MIN = 12.1
    R3_MIN = 12.0
    R4_MIN = 1.0
    R4_MAX = 7.0
    R2_R3_MIN_GAP = 0.1

    # Mesh color limits
    MESH_COLOR_MIN = 108.0
    MESH_COLOR_MAX = 140.0

    # Visual settings
    DEFAULT_TRANSPARENCY = 0.408
    DEFAULT_COLORMAP = "rainbow"
    SMALL_RADIUS = 0.0001
    MEDIUM_RADIUS = 0.0003
    STL_Z_OFFSET = 0.0001

    # Colors (RGB normalized)
    COLOR_BLACK = (0, 0, 0)
    COLOR_GREEN = (0 / 255, 122 / 255, 73 / 255)
    COLOR_LIGHT_BLUE = (157 / 255, 204 / 255, 224 / 255)
    COLOR_DARK_BLUE = (88 / 255, 96 / 255, 143 / 255)
    COLOR_ORANGE = (0.918, 0.545, 0.075)


class VisualizationUI:
    """Handles the user interface elements for the visualization."""

    def __init__(self, visualizer) -> None:
        """Initialize the UI handler with reference to the main visualizer.

        Args:
            visualizer: Reference to the main Visualize instance for accessing data and parameters.

        """
        self.visualizer = visualizer

    def render_controls(self) -> dict[str, bool]:
        """Render all UI controls and return changed parameters."""
        changed = {}

        # Display efficiency
        psim.TextUnformatted(
            f"Efficiency : {self.visualizer.efficiency[0][0]*config.convert_efficiency:.3f} ± "
            f"{abs(self.visualizer.var_efficiency[0][0])*config.convert_efficiency:.3f} %",
        )
        psim.Separator()

        # Point size control
        changed["point_size"], self.visualizer.point_size = psim.SliderFloat(
            "point size",
            self.visualizer.point_size,
            v_min=VisualizationConstants.POINT_SIZE_MIN,
            v_max=VisualizationConstants.POINT_SIZE_MAX,
        )

        # Position controls
        changed["y_move"], self.visualizer.y_move = psim.SliderFloat(
            "Y misalignment (cm)",
            self.visualizer.y_move,
            v_min=VisualizationConstants.POSITION_MIN,
            v_max=VisualizationConstants.POSITION_MAX,
        )
        changed["z_move"], self.visualizer.z_move = psim.SliderFloat(
            "Z misalignment (cm)",
            self.visualizer.z_move,
            v_min=VisualizationConstants.POSITION_MIN,
            v_max=VisualizationConstants.POSITION_MAX,
        )

        # Shape controls
        changed["h1"], self.visualizer.h1 = psim.SliderFloat(
            "h1",
            self.visualizer.h1,
            v_min=VisualizationConstants.H1_MIN,
            v_max=VisualizationConstants.H1_MAX,
        )
        changed["h2"], self.visualizer.h2 = psim.SliderFloat(
            "h2",
            self.visualizer.h2,
            v_min=VisualizationConstants.H2_MIN,
            v_max=VisualizationConstants.H2_MAX,
        )
        changed["h3"], self.visualizer.h3 = psim.SliderFloat(
            "h3",
            self.visualizer.h3,
            v_min=VisualizationConstants.H3_MIN,
            v_max=VisualizationConstants.H3_MAX,
        )
        changed["r1"], self.visualizer.r1 = psim.SliderFloat(
            "r1",
            self.visualizer.r1,
            v_min=VisualizationConstants.R1_MIN,
            v_max=VisualizationConstants.R1_MAX,
        )
        changed["r2"], self.visualizer.r2 = psim.SliderFloat(
            "r2",
            self.visualizer.r2,
            v_min=VisualizationConstants.R2_MIN,
            v_max=self.visualizer.r1,
        )
        changed["r3"], self.visualizer.r3 = psim.SliderFloat(
            "r3",
            self.visualizer.r3,
            v_min=VisualizationConstants.R3_MIN,
            v_max=self.visualizer.r2,
        )
        changed["r4"], self.visualizer.r4 = psim.SliderFloat(
            "r4",
            self.visualizer.r4,
            v_min=VisualizationConstants.R4_MIN,
            v_max=VisualizationConstants.R4_MAX,
        )

        # Vector control
        changed["vector_is_enable"], self.visualizer.vector_is_enable = psim.Checkbox(
            "Vector",
            self.visualizer.vector_is_enable,
        )

        # Calculate weight button
        if psim.Button("calculate weight"):
            changed["calculate_weight"] = True

        return changed


class Visualize:
    """Interactive 3D visualization system for electromagnetic field prediction and ferrite shield optimization.

    This class provides a complete visualization pipeline that combines:
    - 3D geometric modeling of ferrite shields and coils
    - Machine learning-based electromagnetic field predictions
    - Real-time interactive parameter adjustment
    - Performance monitoring and measurement capabilities

    The system uses Gaussian Process models to predict electromagnetic field magnitude,
    direction (theta, phi), and power transfer efficiency based on shield geometry
    and positioning parameters.

    Key features:
    - Interactive 3D visualization using Polyscope
    - Real-time updates when geometry parameters change
    - Linear blend skinning for smooth deformation visualization
    - Color-mapped field magnitude and vector visualization
    - Performance timing and measurement tools

    Args:
        number: Shield configuration number to load from the dataset

    Attributes:
        number: Shield configuration identifier
        ui: User interface handler instance
        point_size: Size of rendered points in 3D view
        y_move, z_move: RX coil displacement parameters
        efficiency: Predicted power transfer efficiency
        values: Predicted electromagnetic field magnitudes
        magnetic_vector: Predicted field direction vectors

    """

    def __init__(self, number: int):
        """Initialize the visualization system."""
        self.number = number

        # Initialize all subsystems
        self._init_visualization_params()
        self._init_geometry_params()
        self._init_ml_params()
        self._init_ui_limits()

        # Initialize UI handler
        self.ui = VisualizationUI(self)

        # Performance monitoring
        self.time_buf = [[0], [0]]
        self._vector_cache = None
        self._vector_needs_update = True

        # Setup the visualization pipeline
        try:
            self._setup_visualization_pipeline()
        except Exception as e:
            logger.error(f"Failed to initialize visualization for shield {number}: {e}")
            raise

    def _init_visualization_params(self):
        """Initialize visualization-related parameters."""
        self.point_size = VisualizationConstants.DEFAULT_POINT_SIZE
        self.mycolor = VisualizationConstants.DEFAULT_COLORMAP
        self.vector_is_enable = False
        self.mesh_color_min = VisualizationConstants.MESH_COLOR_MIN
        self.mesh_color_max = VisualizationConstants.MESH_COLOR_MAX

    def _init_geometry_params(self):
        """Initialize geometry and position parameters."""
        self.y_move = 0.0
        self.z_move = 0.0

        # 3D geometry data
        self.points_for_3D_TX = None
        self.points_for_3D_RX = None
        self.points_for_3D_RX_original = None
        self.lines_for_3D_TX = None
        self.planes_for_3D_TX = None
        self.lines_for_3D_RX = None
        self.planes_for_3D_RX = None
        self.points_for_3D_prediction = None
        self.points_for_3D_prediction_move = None

        # Linear blend skinning
        self.linearblend_skining_weight = None

        # Mesh data
        self.vertices = None
        self.faces = None
        self.vertices_RX_original = None
        self.coil_vertices = None
        self.coil_faces = None
        self.coil_vertices_RX_original = None

        # Grid and prediction data
        self.out_points = None
        self.boxes = None
        self.tetra = None

    def _init_ml_params(self):
        """Initialize machine learning related parameters."""
        # Prediction values
        self.values = None
        self.db_values = None
        self.efficiency = None
        self.var_efficiency = None
        self.magnetic_vector = None

        # Training data and models
        self.X_test = []
        self.X_train = None
        self.K_inv_Y_train_mag = None
        self.K_inv_Y_train_theta = None
        self.K_inv_Y_train_phi = None
        self.lengthscale = None
        self.scale = None
        self.K_inv = None
        self.mean_constant_mag = None
        self.mean_constant_theta = None
        self.mean_constant_phi = None

        # Efficiency models
        self.X_train_efficiency = None
        self.K_inv_Y_train_efficiency = None
        self.lengthscale_efficiency = None
        self.scale_efficiency = None
        self.K_inv_efficiency = None
        self.mean_constant_efficiency = None
        self.noise_efficiency = None

    def _init_ui_limits(self):
        """Initialize UI parameter limits (for backward compatibility)."""
        # Point size limits
        self.point_size_min = VisualizationConstants.POINT_SIZE_MIN
        self.point_size_max = VisualizationConstants.POINT_SIZE_MAX

        # Position limits
        self.y_move_min = self.z_move_min = VisualizationConstants.POSITION_MIN
        self.y_move_max = self.z_move_max = VisualizationConstants.POSITION_MAX

        # Ferrite shield limits
        self.h1_min = VisualizationConstants.H1_MIN
        self.h2_min = self.h3_min = VisualizationConstants.H2_MIN
        self.h1_max = self.h2_max = self.h3_max = VisualizationConstants.H1_MAX
        self.r1_min = self.r2_min = VisualizationConstants.R1_MIN
        self.r1_max = VisualizationConstants.R1_MAX
        self.r3_min = VisualizationConstants.R3_MIN
        self.r4_min = VisualizationConstants.R4_MIN
        self.r4_max = VisualizationConstants.R4_MAX

    def _setup_visualization_pipeline(self):
        """Set up the complete visualization pipeline."""
        logger.debug(f"Initializing visualization pipeline for shield {self.number}")

        # Polyscope objects (initialized later)
        self.TX_point = None
        self.TX_edge = None
        self.TX_plane = None
        self.RX_point = None
        self.RX_edge = None
        self.RX_plane = None
        self.coil_RX = None
        self.ps_mesh_RX = None
        self.Prediction_point = None

        # Create prediction structure
        self.make_prediction_structure()

        # Setup ferrite shield parameters
        self.make_ferrite_shield()

        # Prepare ML predictions
        self.prediction_prepare()

    def visualize(
        self, measure_time_mode: bool = False, timeout_seconds: int = 0
    ) -> None:
        """Launch the interactive 3D visualization window.

        Initialize Polyscope 3D viewer and display the electromagnetic field visualization
        with interactive controls. The visualization includes ferrite shields, coils,
        prediction points, and real-time field magnitude/vector displays.

        The right window allows you to edit the coil's position and shape while viewing the magnetic field strength and power transfer efficiency in real time. For more detailed instructions, please refer to the Readme.

        Args:
            measure_time_mode: If True, collect and save timing measurements for
                             performance analysis to time_Level2.txt file.
            timeout_seconds: If > 0, automatically close window after specified seconds.
                            If 0, window stays open until manually closed.

        Note:
            This function corresponds to Figure 1a and Table 1 in the paper.

        """
        # ---------------------------可視化用---------------------------------
        # Polyscopeを初期化
        logger.info("Initializing Polyscope 3D visualization...")
        logger.debug(f"Visualization parameters - Shield number: {self.number}")
        logger.debug(
            f"Performance monitoring: {'enabled' if measure_time_mode else 'disabled'}"
        )
        if timeout_seconds > 0:
            logger.debug(f"Auto-close timeout: {timeout_seconds} seconds")

        ps.init()
        logger.debug("Polyscope backend initialized successfully")

        logger.debug("Preparing visualization geometry and UI components")
        self.visualization_prepare()
        logger.info("Visualization preparation completed")

        logger.debug("Setting up interactive UI callback")
        ps.set_user_callback(self.callback)

        # 初期の視点を指定
        ps.set_up_dir("z_up")
        # 地面の設定で、地面をなしにしている
        ps.set_ground_plane_mode("none")

        # 描画
        logger.info("Opening interactive 3D visualization window...")

        # Handle timeout if specified
        if timeout_seconds > 0:

            def close_after_timeout():
                time.sleep(timeout_seconds)
                logger.info(
                    f"Auto-closing visualization window after {timeout_seconds} seconds"
                )
                ps.shutdown()

            # Start timeout thread
            timeout_thread = threading.Thread(target=close_after_timeout, daemon=True)
            timeout_thread.start()

        ps.show()

        if measure_time_mode:
            with (paths.RESULT_MEASURE_TIME / "time_Level2.txt").open("w") as f:
                self.time_buf[0] = np.delete(self.time_buf[0], [0, 1], 0)
                self.time_buf[1] = np.delete(self.time_buf[1], [0, 1], 0)
                print("change position", file=f)
                print(
                    f"average : {np.mean(self.time_buf[0])}, number: {len(self.time_buf[0])}",
                    file=f,
                )
                for i in range(len(self.time_buf[0])):
                    print(self.time_buf[0][i], file=f)
                print("change shape", file=f)
                print(
                    f"average : {np.mean(self.time_buf[1])}, number: {len(self.time_buf[1])}",
                    file=f,
                )
                for i in range(len(self.time_buf[1])):
                    print(self.time_buf[1][i], file=f)

    def callback(self):
        """Handle all user interactions in the main UI callback function."""
        start = time.time()

        try:
            # Render UI controls and get changed parameters
            changed = self.ui.render_controls()

            # Handle different types of changes
            self._handle_point_size_change(changed)
            self._handle_position_changes(changed, start)
            self._handle_shape_changes(changed, start)
            self._handle_weight_calculation(changed)

        except Exception as e:
            logger.error(f"Error in UI callback: {e}")

    def _handle_point_size_change(self, changed: dict):
        """Handle point size changes."""
        if changed.get("point_size") and self.Prediction_point:
            self.Prediction_point.set_radius(self.point_size / 1e3)

    def _handle_position_changes(self, changed: dict, start_time: float):
        """Handle position-related changes (y_move, z_move, vector display)."""
        position_changed = changed.get("y_move") or changed.get("z_move")
        vector_changed = changed.get("vector_is_enable")

        if position_changed:
            self._update_rx_positions()

        if position_changed or vector_changed:
            self._update_prediction_values_for_move()
            self._update_visualization_window()
            self._update_vector_display(vector_changed)
            self._measure_finish_time(start_time, 0)  # 0 for position changes

    def _handle_shape_changes(self, changed: dict, start_time: float):
        """Handle ferrite shield shape changes."""
        shape_params = ["h1", "h2", "h3", "r1", "r2", "r3", "r4"]
        shape_changed = any(changed.get(param) for param in shape_params)

        if shape_changed:
            # Update shape-specific parameters
            self.update_values_for_shape(
                changed.get("h1", False),
                changed.get("h2", False),
                changed.get("h3", False),
                changed.get("r1", False),
                changed.get("r2", False),
                changed.get("r3", False),
                changed.get("r4", False),
            )

            # Update deformation
            Deformation.set_points(
                self.TX_point,
                self.TX_edge,
                self.TX_plane,
                self.RX_point,
                self.RX_edge,
                self.RX_plane,
                self.points_for_3D_TX,
                self.points_for_3D_RX,
            )
            Deformation.move_out_points(
                self.out_points,
                self.points_for_3D_TX,
                self.y_move,
                self.z_move,
                self.points_for_3D_prediction_move,
                self.linearblend_skining_weight,
                self.points_for_3D_RX_original,
            )

            self._update_visualization_window()
            self._measure_finish_time(start_time, 1)  # 1 for shape changes

    def _handle_weight_calculation(self, changed: dict):
        """Handle weight recalculation."""
        if changed.get("calculate_weight"):
            self.linearblend_skining_weight = Visualize.calculate_weight(
                self.points_for_3D_TX,
                self.points_for_3D_prediction,
            )

    def _update_rx_positions(self):
        """Update RX coil and shield positions based on y_move and z_move."""
        offset = np.array(
            [
                0.0,
                self.y_move / config.meter_to_centimeter,
                self.z_move / config.meter_to_centimeter,
            ],
        )

        # Update RX points
        self.points_for_3D_RX = self.points_for_3D_RX_original + offset

        # Update coil vertices
        coil_vertices_rx = self.coil_vertices_RX_original + offset

        # Update shield vertices
        vertices_rx = self.vertices_RX_original + offset

        # Update Polyscope objects
        if self.RX_point:
            self.RX_point.update_point_positions(self.points_for_3D_RX)
        if self.RX_edge:
            self.RX_edge.update_node_positions(self.points_for_3D_RX)
        if self.RX_plane:
            self.RX_plane.update_vertex_positions(self.points_for_3D_RX)
        if self.coil_RX:
            self.coil_RX.update_vertex_positions(coil_vertices_rx)
        if self.ps_mesh_RX:
            self.ps_mesh_RX.update_vertex_positions(vertices_rx)

    def _update_prediction_values_for_move(self):
        """Update prediction values when position changes."""
        self.update_values_for_move()

    def _update_visualization_window(self):
        """Update the visualization window with new values."""
        self.update_window()

    def _update_vector_display(self, vector_changed: bool) -> None:
        """Update magnetic vector display with caching for performance."""
        if self.vector_is_enable or vector_changed:
            # Use cached vector if available and not outdated
            if self._vector_cache is None or self._vector_needs_update:
                self._vector_cache = Prediction.vector(
                    self.X_train,
                    self.X_test,
                    self.K_inv_Y_train_theta,
                    self.K_inv_Y_train_phi,
                    self.lengthscale,
                    self.scale,
                    self.mean_constant_theta,
                    self.mean_constant_phi,
                )[0]
                self._vector_needs_update = False

            if self.Prediction_point:
                self.Prediction_point.add_vector_quantity(
                    "my vectors",
                    self._vector_cache,
                    enabled=self.vector_is_enable,
                )

    def _measure_finish_time(self, start_time: float, buffer_index: int):
        """Measure and record execution time."""
        end_time = time.time()
        time_diff = end_time - start_time
        self.time_buf[buffer_index].append(time_diff)

    @staticmethod
    def make_modeling_visualization(
        input_path: Path,
        y_move: float,
        z_move: float,
    ) -> tuple:
        """Generate 3D geometric data structures for visualization of ferrite shield system.

        Create hierarchical grid structures for TX/RX coils and prediction points using
        multi-level refinement. Generates points, edges, planes, and tetrahedral elements
        for 3D visualization and electromagnetic field prediction.

        When expressing a shape, a grid structure starts with a coarse grid and is refined in stages.
        In addition, the prediction points are created to match the shape's grid, and these too are
        refined to align with the points on the surface. While a finer grid allows for a better
        representation of the shape, it increases computation time, which can compromise real-time prediction.

        Args:
            input_path: Path to input data file containing initial grid configuration
            y_move: Y-axis displacement of RX coil relative to TX coil (cm)
            z_move: Z-axis displacement of RX coil relative to TX coil (cm)

        Returns:
            tuple: Contains 8 elements:
                - points_for_3D_TX: TX coil point coordinates
                - lines_for_3D_TX: TX coil edge connectivity
                - planes_for_3D_TX: TX coil surface triangulation
                - points_for_3D_RX: RX coil point coordinates (displaced)
                - points_for_3D_prediction_move: Prediction points after linear blend skinning
                - points_for_3D_prediction: Original prediction point positions
                - out_points: Grid output points with mode information
                - boxes_for_3D_prediction: Tetrahedral boxes for color mapping

        Note:
            Uses 3-level hierarchical refinement (level1 -> level2 -> level3) for
            improved geometric accuracy and prediction point density.
            This function corresponds to Figure 2, Figure 3, Figure 4, Supplementary Figure 2, Supplementary Figure 5, Supplementary Figure 6, and Supplementary Figure 7 in the paper.

        """
        # Initialize TX points and mesh structure (level1)
        points, edges, planes = make_first_point_edge_plane(input_path)
        # Initialize prediction point positions (level1)
        out_points, boxes, edges_outside, planes_outside = make_first_out_points(
            points,
            planes,
        )
        # Refine TX points and mesh structure (level2)
        points, edges, planes = make_second_grid(input_path, points, edges, planes)
        # Refine prediction point positions (level2)
        out_points, boxes, edges_outside, planes_outside = make_second_out_points(
            out_points,
            boxes,
            edges_outside,
            planes_outside,
            points,
            edges,
            planes,
        )
        # Further refine TX points and mesh structure (level3)
        points, edges, planes = make_second_grid(input_path, points, edges, planes)

        # データの可視化のためのTXの点の位置を保存するための配列
        points_for_3d_tx = np.array([points[i].xyz for i in range(len(points))])

        # データの可視化のためのRXの点の位置を保存するための配列
        # 位置に関してはRXはTXをz = 0.0025で対照に写したものをy,z方向に動かす
        points_for_3d_rx = np.array(
            [
                [
                    points[i].xyz[0],
                    points[i].xyz[1] + y_move,
                    config.initial_coil_distance - points[i].xyz[2] + z_move,
                ]
                for i in range(len(points))
            ],
        )

        # データの可視化のためのTXの辺を保存するための配列
        lines_for_3d_tx = np.empty((0, 2), int)
        for i in range(len(edges)):
            if edges[i].children[0] == -1:
                lines_for_3d_tx = np.append(
                    lines_for_3d_tx,
                    [[edges[i].node[0], edges[i].node[1]]],
                    axis=0,
                )

        # データの可視化のためのTXの平面を保存するための配列
        planes_for_3d_tx = np.empty((0, 3), int)
        for i in range(len(planes)):
            if planes[i].children[0] == -1:
                planes_for_3d_tx = np.append(
                    planes_for_3d_tx,
                    [[planes[i].node[0], planes[i].node[1], planes[i].node[2]]],
                    axis=0,
                )
                planes_for_3d_tx = np.append(
                    planes_for_3d_tx,
                    [[planes[i].node[2], planes[i].node[3], planes[i].node[0]]],
                    axis=0,
                )

        # データの可視化のための予測の点の位置を保存するための配列
        points_for_3d_prediction = np.empty((0, 3), float)
        poplist = []
        for i in range(len(out_points)):
            # TXやRXの表面にあるものは除外
            if out_points[i].mode == 0:
                points_for_3d_prediction = np.append(
                    points_for_3d_prediction,
                    [out_points[i].xyz],
                    axis=0,
                )
            else:
                poplist.append(i)
        points_for_3d_prediction_move = Visualize.linearblendskining(
            points_for_3d_tx,
            points_for_3d_rx,
            points_for_3d_prediction.copy(),
        )

        # カラーマップの作成のために予測点の位置に対するboxの作成
        logger.debug(
            "Processing prediction points and tetrahedral boxes for color mapping"
        )
        boxes_for_3d_prediction = []
        for i in range(len(boxes)):
            box = copy.deepcopy(boxes[i])
            flag = 0
            for j in range(len(box.node)):
                if out_points[box.node[j]].mode != 0:
                    flag = 1
                    break
                index = bisect.bisect(poplist, box.node[j])
                box.node[j] -= index
            if flag == 0:
                boxes_for_3d_prediction.append(box)

        return (
            points_for_3d_tx,
            lines_for_3d_tx,
            planes_for_3d_tx,
            points_for_3d_rx,
            points_for_3d_prediction_move,
            points_for_3d_prediction,
            out_points,
            boxes_for_3d_prediction,
        )

    @staticmethod
    def linearblendskining(
        tx_points: np.array,
        rx_points: np.array,
        prediction_points: np.array,
    ) -> np.array:
        """Apply linear blend skinning deformation to prediction points based on coil movement.

        Deform prediction points based on the transformation between TX and RX coil positions
        using linear blend skinning algorithm. Each prediction point is weighted by its distance
        to ferrite shield control points.

        Args:
            tx_points: Control points on TX (transmitter) ferrite shield
            rx_points: Control points on RX (receiver) ferrite shield (displaced)
            prediction_points: Original prediction point positions to be deformed

        Returns:
            np.array: Deformed prediction points after linear blend skinning transformation

        Note:
            Uses distance-based weighting with configurable power parameter (LBS_w_move).
            This function corresponds to Figure 2 and Figure 3 in the paper.

        """
        weight, rx_ferrite_point = Visualize.calculate_weight(
            tx_points,
            prediction_points,
        )

        # フェライトシールドの各点の変化を計算
        change = np.array([[0.0, 0.0, 0.0] for _ in range(config.n_initial_points * 2)])
        for k in range(config.n_initial_points):
            change[k + config.n_initial_points] = rx_points[k] - rx_ferrite_point[k]

        # prediction_pointにフェライトシールドの変化分を足す
        for j in range(len(prediction_points)):
            for k in range(config.n_initial_points * 2):
                prediction_points[j] += change[k] * weight[j][k]

        return prediction_points

    @staticmethod
    def calculate_weight(tx_points, prediction_points):
        """Calculate linear blend skinning weights for prediction points based on distance to control points.

        Compute normalized weights for each prediction point relative to TX and RX ferrite
        shield control points. Weights are inversely proportional to distance raised to
        a configurable power (LBS_w_move).

        Args:
            tx_points: Control points on TX ferrite shield (n_initial_points x 3)
            prediction_points: Points to calculate weights for (N x 3)

        Returns:
            tuple: (weight, rx_ferrite_point) where:
                - weight: Normalized weights array (N x 2*n_initial_points)
                - rx_ferrite_point: Original RX control point positions before displacement

        Note:
            Uses config.LBS_w_move for distance weighting power and config.n_initial_points
            for the number of control points (typically 72 points per shield).
            This function corresponds to Figure 3 in the paper.

        """
        # TX(送信器)、RX(受信器)のフェライトシールドの点
        # それぞれ72点なのは基準点で、これらのみを元にlinearblendskinningをを行うため
        # RXは元々の位置をまずは格納
        tx_ferrite_point = np.array(
            [[0.0, 0.0, 0.0] for _ in range(config.n_initial_points)],
        )
        rx_ferrite_point = np.array(
            [[0.0, 0.0, 0.0] for _ in range(config.n_initial_points)],
        )
        for j in range(config.n_initial_points):
            tx_ferrite_point[j][0] = tx_points[j][0]
            tx_ferrite_point[j][1] = tx_points[j][1]
            tx_ferrite_point[j][2] = tx_points[j][2]
            rx_ferrite_point[j][0] = tx_points[j][0]
            rx_ferrite_point[j][1] = tx_points[j][1]
            rx_ferrite_point[j][2] = config.initial_coil_distance - tx_points[j][2]

        # linearblendskiningで用いる変数
        distance_weight = config.LBS_w_move

        # 予測点それぞれに対してすべての基準点の重みを格納
        weight = np.zeros((len(prediction_points), config.n_initial_points * 2))
        for j in range(len(prediction_points)):

            weight_sum = 0  #

            # それぞれのフェライトの点に対して重みを計算して合計する
            for k in range(config.n_initial_points * 2):
                if k < config.n_initial_points:
                    d = np.linalg.norm(
                        tx_ferrite_point[k] - prediction_points[j],
                        ord=2,
                    )
                    weight[j][k] = 1 / d ** (distance_weight)
                    weight_sum += 1 / d ** (distance_weight)
                else:
                    d = np.linalg.norm(
                        rx_ferrite_point[k - config.n_initial_points]
                        - prediction_points[j],
                        ord=2,
                    )
                    weight[j][k] = 1 / d ** (distance_weight)
                    weight_sum += 1 / d ** (distance_weight)

            # それぞれのフェライトの点に対する重みを正規化して、本当の重みとして保存する
            weight[j] /= weight_sum

        return weight, rx_ferrite_point

    @staticmethod
    def make_tetra(boxes):
        """Convert hexahedral boxes to tetrahedral elements for volume mesh visualization.

        Decompose each hexahedral box into 6 tetrahedral elements for use in Polyscope
        volume mesh visualization. Each box is split using a consistent pattern to
        maintain mesh continuity.

        Using this, the visualization tool can visualize the magnetic field distribution on a plane.

        Args:
            boxes: List of box objects with node indices defining hexahedral elements

        Returns:
            np.array: Tetrahedral connectivity array (N*6 x 4) where each row contains
                     4 node indices defining a tetrahedron

        Note:
            tetrahedralization pattern used for consistency verification.

        """
        tetra = []
        for i in range(len(boxes)):
            box = boxes[i]
            tet = [
                [box.node[0], box.node[1], box.node[3], box.node[7]],
                [box.node[0], box.node[1], box.node[5], box.node[7]],
                [box.node[0], box.node[2], box.node[3], box.node[7]],
                [box.node[0], box.node[2], box.node[6], box.node[7]],
                [box.node[0], box.node[4], box.node[5], box.node[7]],
                [box.node[0], box.node[4], box.node[6], box.node[7]],
            ]
            for j in range(6):
                tetra.append(tet[j])
        return np.array(tetra)

    @staticmethod
    def convert_to_db(values):
        """Convert magnetic field values to decibel scale (dBµA/m).

        Transform mu_star values to decibel microamperes per meter for visualization
        and analysis purposes.

        Args:
            values: Magnetic field values in original units

        Returns:
            Converted values in dBµA/m scale

        Note:
            of mu_star and the conversion formula derivation.

        """
        # mu_starをdBuA/mに変換
        return 20 * np.log10(values * 10**6)

    def measure_finish_time(self, start):
        """Use _measure_finish_time instead (deprecated method)."""
        logger.warning("measureFinishTime is deprecated, use _measure_finish_time")
        self._measure_finish_time(start, 1)

    def update_window(self):
        """Update the 3D visualization window with current prediction values and color mapping.

        Refresh the Polyscope visualization by updating prediction point positions and
        scalar field values. Creates new volume mesh with current dB values and applies
        color mapping within specified min/max bounds.

        Note:
            and add error handling for visualization updates.

        """
        self.Prediction_point.update_point_positions(self.points_for_3D_prediction_move)
        self.Prediction_point.add_scalar_quantity("my values", self.db_values)
        ps_mesh = ps.register_volume_mesh(
            "Color Map",
            self.points_for_3D_prediction_move,
            self.tetra,
        )
        ps_mesh.add_scalar_quantity(
            "My Scalar Quantity",
            self.db_values,
            defined_on="vertices",
            vminmax=(self.mesh_color_min, self.mesh_color_max),
            cmap=self.mycolor,
            enabled=True,
        )

    def make_prediction_structure(self):
        """Initialize 3D geometric structures for electromagnetic field prediction.

        Create the complete 3D modeling framework including TX/RX coil geometries,
        prediction point grids, and linear blend skinning weights. Loads data from
        the circle_move_x dataset and sets up all geometric relationships needed
        for field prediction and visualization.

        Sets up:
            - TX/RX coil point coordinates, edges, and planes
            - Prediction point grid with linear blend skinning weights
            - Tetrahedral boxes for volume mesh color mapping

        Note:
            focused methods for better maintainability.

        """
        logger.info(f"Starting 3D modeling for shield number {self.number}")
        input_path_x_data = paths.CIRCLE_MOVE_X_DIR / template.circle_move_x.format(
            index=self.number,
        )
        grid_info = Visualize.make_modeling_visualization(
            input_path_x_data,
            self.y_move,
            self.z_move,
        )

        # TXの点、辺、平面の情報
        self.points_for_3D_TX = grid_info[0]
        self.lines_for_3D_TX = grid_info[1]
        self.planes_for_3D_TX = grid_info[2]

        # RXの点、辺、平面の情報(辺と平面の情報はTXと同じ)
        self.points_for_3D_RX = grid_info[3]
        self.lines_for_3D_RX = grid_info[1]
        self.planes_for_3D_RX = grid_info[2]

        # point更新のためのoriginalを用意
        self.points_for_3D_RX_original = self.points_for_3D_RX.copy()

        # 予測点の点、辺の情報
        self.points_for_3D_prediction_move = grid_info[4]

        # weightのための元々の予測点の位置
        self.points_for_3D_prediction = grid_info[5]

        # linearblendskiningのweight
        self.linearblend_skining_weight, _ = Visualize.calculate_weight(
            self.points_for_3D_TX,
            self.points_for_3D_prediction,
        )

        # 予測点の位置
        self.out_points = grid_info[6]

        # カラーマップのためのbox
        self.boxes = grid_info[7]

    def make_ferrite_shield(self):
        """Initialize ferrite shield geometric parameters and constraints.

        Set up all ferrite shield dimensional parameters (h1, h2, h3, r1, r2, r3, r4)
        and their allowable ranges for optimization. Uses the Deformation module to
        prepare ferrite shield movement and deformation capabilities.

        Initializes:
            - Shield height parameters (h1, h2, h3) and their ranges
            - Shield radius parameters (r1, r2, r3, r4) and constraints
            - Parameter lists for UI controls and optimization bounds

        Note:
            meaning of each parameter (h1=inner height, r1=outer radius, etc.).
            This function corresponds to Supplementary Figure 1 in the paper.

        """
        (
            self.h1_low,
            self.h1_middle,
            self.h2_list,
            self.h3_list,
            self.r1_list,
            self.r2_list,
            self.r3_list,
            self.r2_r3_list,
            self.r4_list,
            self.h1,
            self.h2,
            self.h3,
            self.r1,
            self.r2,
            self.r3,
            self.r4,
        ) = Deformation.prepare_ferrite_move(self.points_for_3D_TX)

    def prediction_prepare(self):
        """Prepare ML models and initial predictions with comprehensive error handling."""
        try:

            # Load main prediction models
            prediction_data = Prediction.prediction_prepare()
            if len(prediction_data) != 10:
                error_msg = (
                    f"Expected 10 prediction parameters, got {len(prediction_data)}"
                )
                raise ValueError(error_msg)

            (
                self.X_train,
                self.K_inv_Y_train_mag,
                self.K_inv_Y_train_theta,
                self.K_inv_Y_train_phi,
                self.lengthscale,
                self.scale,
                self.K_inv,
                self.mean_constant_mag,
                self.mean_constant_theta,
                self.mean_constant_phi,
            ) = prediction_data

            logger.debug("Loading efficiency models...")

            # Load efficiency models
            efficiency_data = Prediction.prediction_prepare_efficiency()
            if len(efficiency_data) != 7:
                error_msg = (
                    f"Expected 7 efficiency parameters, got {len(efficiency_data)}"
                )
                raise ValueError(error_msg)

            (
                self.X_train_efficiency,
                self.K_inv_Y_train_efficiency,
                self.lengthscale_efficiency,
                self.scale_efficiency,
                self.K_inv_efficiency,
                self.mean_constant_efficiency,
                self.noise_efficiency,
            ) = efficiency_data

            # Prepare test data
            self._prepare_test_data()

            # Run initial predictions
            self._run_initial_predictions()

            logger.debug("ML prediction preparation completed successfully")

        except FileNotFoundError as e:
            logger.error(f"Model files not found: {e}")
            msg = f"Failed to load ML models - missing files: {e}"
            raise RuntimeError(msg) from e
        except ValueError as e:
            logger.error(f"Invalid model data format: {e}")
            msg = f"Model data format error: {e}"
            raise RuntimeError(msg) from e
        except Exception as e:
            logger.error(f"Unexpected error during prediction preparation: {e}")
            msg = f"Failed to prepare ML predictions: {e}"
            raise RuntimeError(msg) from e

    def _prepare_test_data(self):
        """Prepare test data from 3D TX points and movement parameters."""
        if self.points_for_3D_TX is None:
            error_msg = "3D TX points not initialized"
            raise ValueError(error_msg)

        self.X_test = []
        for i in range(len(self.points_for_3D_TX)):
            for j in range(3):
                self.X_test.append(self.points_for_3D_TX[i][j])

        self.X_test.append(self.y_move)
        self.X_test.append(self.z_move)
        self.X_test = np.array([self.X_test])

        logger.debug(f"Prepared test data with shape: {self.X_test.shape}")

    def _run_initial_predictions(self):
        """Run initial ML predictions for the current configuration."""
        try:
            logger.debug("Running ML predictions for current shield configuration...")

            # Predict magnetic field magnitude
            self.values = Prediction.mu_star(
                self.X_train,
                self.X_test,
                self.K_inv_Y_train_mag,
                self.lengthscale,
                self.scale,
                self.mean_constant_mag,
            ).reshape(-1)

            # Convert to dB scale
            self.db_values = Visualize.convert_to_db(self.values)

            # Predict magnetic field vector
            self.magnetic_vector = Prediction.vector(
                self.X_train,
                self.X_test,
                self.K_inv_Y_train_theta,
                self.K_inv_Y_train_phi,
                self.lengthscale,
                self.scale,
                self.mean_constant_theta,
                self.mean_constant_phi,
            )[0]

            # Predict efficiency
            self.efficiency, self.var_efficiency = Prediction.mu_star_efficiency(
                self.X_train_efficiency,
                self.X_test,
                self.K_inv_Y_train_efficiency,
                self.lengthscale_efficiency,
                self.scale_efficiency,
                self.K_inv_efficiency,
                self.mean_constant_efficiency,
                self.noise_efficiency,
            )

            # Mark vector cache as valid
            self._vector_cache = self.magnetic_vector
            self._vector_needs_update = False

            logger.info(
                f"Predictions completed - Efficiency: {self.efficiency[0][0]*config.convert_efficiency:.3f}%",
            )

        except Exception as e:
            logger.error(f"Failed to run initial predictions: {e}")
            raise

    def visualization_prepare(self):
        """Prepare all Polyscope visualization objects with error handling."""
        try:
            logger.debug("Preparing 3D visualization objects...")

            # Load and prepare STL meshes
            self._load_stl_meshes()

            # Setup TX (transmitter) visualization objects
            self._setup_tx_visualization()

            # Setup RX (receiver) visualization objects
            self._setup_rx_visualization()

            # Setup prediction points visualization
            self._setup_prediction_visualization()

            # Setup volume mesh for color mapping
            self._setup_volume_mesh()

            # Setup coil visualization
            self._setup_coil_visualization()

            logger.info("3D visualization preparation completed successfully")

        except Exception as e:
            logger.error(f"Failed to prepare visualization: {e}")
            msg = f"Visualization preparation failed: {e}"
            raise RuntimeError(msg) from e

    def _load_stl_meshes(self):
        """Load STL mesh files for shields and coils."""
        try:
            # Load ferrite shield STL
            input_path_stl = paths.CIRCLE_STL_DIR / template.circle_stl.format(
                index=self.number,
            )
            if not input_path_stl.exists():
                msg = f"Shield STL file not found: {input_path_stl}"
                raise FileNotFoundError(msg)

            shield_mesh = mesh.Mesh.from_file(input_path_stl)
            self.vertices = np.reshape(shield_mesh.vectors, (-1, 3))
            self.faces = np.reshape(
                np.arange(len(self.vertices)).reshape((-1, 3)),
                (-1, 3),
            )

            # Apply Z-offset to prevent rendering issues
            self.vertices = np.array(
                [
                    [
                        vertex[0],
                        vertex[1],
                        vertex[2] - VisualizationConstants.STL_Z_OFFSET,
                    ]
                    for vertex in self.vertices
                ],
            )

            # Create RX vertices (mirrored and offset)
            self.vertices_RX_original = np.array(
                [
                    [
                        vertex[0],
                        vertex[1] + self.y_move / config.meter_to_centimeter,
                        config.initial_coil_distance
                        - vertex[2]
                        + self.z_move / config.meter_to_centimeter
                        + VisualizationConstants.STL_Z_OFFSET,
                    ]
                    for vertex in self.vertices
                ],
            )

            # Load coil STL
            input_path_coil_stl = paths.CIRCLE_STL_DIR / "coil.stl"
            if not input_path_coil_stl.exists():
                msg = f"Coil STL file not found: {input_path_coil_stl}"
                raise FileNotFoundError(
                    msg,
                )

            coil_mesh = mesh.Mesh.from_file(input_path_coil_stl)
            self.coil_vertices = np.reshape(coil_mesh.vectors, (-1, 3))
            self.coil_faces = np.reshape(
                np.arange(len(self.coil_vertices)).reshape((-1, 3)),
                (-1, 3),
            )

        except Exception as e:
            logger.error(f"Failed to load STL meshes: {e}")
            raise

    def _setup_tx_visualization(self):
        """Set up TX (transmitter) visualization objects."""
        self.TX_point = ps.register_point_cloud(
            "TX points", self.points_for_3D_TX, enabled=False
        )
        self.TX_edge = ps.register_curve_network(
            "TX edges",
            self.points_for_3D_TX,
            self.lines_for_3D_TX,
        )
        self.TX_plane = ps.register_surface_mesh(
            "TX planes",
            self.points_for_3D_TX,
            self.planes_for_3D_TX,
        )

        # Apply TX styling
        self.TX_point.set_radius(VisualizationConstants.SMALL_RADIUS, relative=False)
        self.TX_point.set_color(VisualizationConstants.COLOR_BLACK)
        self.TX_edge.set_radius(VisualizationConstants.MEDIUM_RADIUS)
        self.TX_edge.set_color(VisualizationConstants.COLOR_GREEN)
        self.TX_plane.set_color(VisualizationConstants.COLOR_LIGHT_BLUE)

    def _setup_rx_visualization(self):
        """Set up RX (receiver) visualization objects."""
        self.RX_point = ps.register_point_cloud(
            "RX points", self.points_for_3D_RX, enabled=False
        )
        self.RX_edge = ps.register_curve_network(
            "RX edges",
            self.points_for_3D_RX,
            self.lines_for_3D_RX,
        )
        self.RX_plane = ps.register_surface_mesh(
            "RX planes",
            self.points_for_3D_RX,
            self.planes_for_3D_RX,
        )

        # Apply RX styling (same as TX)
        self.RX_point.set_radius(VisualizationConstants.SMALL_RADIUS, relative=False)
        self.RX_point.set_color(VisualizationConstants.COLOR_BLACK)
        self.RX_edge.set_radius(VisualizationConstants.MEDIUM_RADIUS)
        self.RX_edge.set_color(VisualizationConstants.COLOR_GREEN)
        self.RX_plane.set_color(VisualizationConstants.COLOR_LIGHT_BLUE)

    def _setup_prediction_visualization(self):
        """Set up prediction points visualization."""
        self.Prediction_point = ps.register_point_cloud(
            "Prediction points",
            self.points_for_3D_prediction_move,
        )
        self.Prediction_point.add_scalar_quantity(
            "my values",
            self.db_values,
            vminmax=(self.mesh_color_min, self.mesh_color_max),
            cmap=self.mycolor,
            enabled=True,
        )
        self.Prediction_point.set_radius(self.point_size / 1e3)

        # Add vector quantity (initially disabled)
        self.Prediction_point.add_vector_quantity(
            "my vectors",
            self.magnetic_vector,
            enabled=False,
        )

    def _setup_volume_mesh(self):
        """Set up volume mesh for color mapping."""
        self.tetra = Visualize.make_tetra(self.boxes)
        ps_mesh = ps.register_volume_mesh(
            "Color Map",
            self.points_for_3D_prediction_move,
            self.tetra,
            enabled=False,
        )
        ps_mesh.add_scalar_quantity(
            "My Scalar Quantity",
            self.db_values,
            defined_on="vertices",
            vminmax=(self.mesh_color_min, self.mesh_color_max),
            cmap=self.mycolor,
            enabled=True,
        )

    def _setup_coil_visualization(self):
        """Set up coil visualization objects."""
        # TX coil
        coil_tx = ps.register_surface_mesh(
            "coil_TX",
            self.coil_vertices,
            self.coil_faces,
        )
        coil_tx.set_color(VisualizationConstants.COLOR_ORANGE)

        # RX coil (mirrored and positioned)
        self.coil_vertices_RX = (
            self.coil_vertices * np.array([1.0, 1.0, -1.0])
            + np.array([0.0, 0.0, config.initial_coil_distance])
            + np.array(
                [
                    0.0,
                    self.y_move / config.meter_to_centimeter,
                    self.z_move / config.meter_to_centimeter,
                ],
            )
        )
        self.coil_vertices_RX_original = self.coil_vertices_RX.copy()

        self.coil_RX = ps.register_surface_mesh(
            "coil_RX",
            self.coil_vertices_RX,
            self.coil_faces,
        )
        self.coil_RX.set_color(VisualizationConstants.COLOR_ORANGE)

    def update_values_for_move(self):
        """Update prediction values when position changes."""
        # Update prediction point positions using linear blend skinning
        change_yz = np.array(
            [[0.0, 0.0, 0.0] for _ in range(config.n_initial_points * 2)],
        )
        for k in range(config.n_initial_points):
            change_yz[k + config.n_initial_points] = np.array(
                [
                    0.0,
                    self.y_move / config.meter_to_centimeter,
                    self.z_move / config.meter_to_centimeter,
                ],
            )

        for i in range(len(self.points_for_3D_prediction_move)):
            self.points_for_3D_prediction_move[i] = self.points_for_3D_prediction[
                i
            ] + np.dot(self.linearblend_skining_weight[i], change_yz)

        # Update test data with new position
        self.X_test[0][-2] = self.y_move / config.meter_to_centimeter
        self.X_test[0][-1] = self.z_move / config.meter_to_centimeter

        # Recalculate predictions
        self.values = Prediction.mu_star(
            self.X_train,
            self.X_test,
            self.K_inv_Y_train_mag,
            self.lengthscale,
            self.scale,
            self.mean_constant_mag,
        ).reshape(-1)
        self.dBvalues = Visualize.convert_to_db(self.values)

        self.efficiency, self.var_efficiency = Prediction.mu_star_efficiency(
            self.X_train_efficiency,
            self.X_test,
            self.K_inv_Y_train_efficiency,
            self.lengthscale_efficiency,
            self.scale_efficiency,
            self.K_inv_efficiency,
            self.mean_constant_efficiency,
            self.noise_efficiency,
        )

        # Mark vector cache as needing update
        self._vector_needs_update = True

    def update_values_for_shape(
        self,
        changed_h1: bool,
        changed_h2: bool,
        changed_h3: bool,
        changed_r1: bool,
        changed_r2: bool,
        changed_r3: bool,
        changed_r4: bool,
    ):
        """Update ferrite shield geometry based on changed shape parameters.

        Modify the 3D geometry of TX and RX ferrite shields when dimensional parameters
        change. Updates both point positions and mesh geometry, then triggers prediction
        recalculation and visualization refresh.

        Args:
            changed_h1: Whether h1 (height parameter 1) has changed
            changed_h2: Whether h2 (height parameter 2) has changed
            changed_h3: Whether h3 (height parameter 3) has changed
            changed_r1: Whether r1 (radius parameter 1) has changed
            changed_r2: Whether r2 (radius parameter 2) has changed
            changed_r3: Whether r3 (radius parameter 3) has changed
            changed_r4: Whether r4 (radius parameter 4) has changed

        Note:
            multiple boolean flags for better maintainability.

        """
        if changed_h1:
            Deformation.change_h1(
                self.points_for_3D_TX,
                self.points_for_3D_RX,
                self.z_move,
                self.h1 / config.meter_to_centimeter,
                self.h1_low,
                self.h1_middle,
            )
        elif changed_h2:
            Deformation.change_h2(
                self.points_for_3D_TX,
                self.points_for_3D_RX,
                self.z_move,
                self.h2 / config.meter_to_centimeter,
                self.h2_list,
            )
        elif changed_h3:
            Deformation.change_h3(
                self.points_for_3D_TX,
                self.points_for_3D_RX,
                self.z_move,
                self.h3 / config.meter_to_centimeter,
                self.h3_list,
            )
        elif changed_r1:
            Deformation.change_r1(
                self.points_for_3D_TX,
                self.points_for_3D_RX,
                self.y_move,
                self.r1 / config.meter_to_centimeter,
                self.r1_list,
            )
        elif changed_r2:
            if self.r2 - VisualizationConstants.R2_R3_MIN_GAP <= self.r3:
                self.r3 = self.r2 - VisualizationConstants.R2_R3_MIN_GAP
                Deformation.change_r3(
                    self.points_for_3D_TX,
                    self.points_for_3D_RX,
                    self.y_move,
                    self.r2 / config.meter_to_centimeter,
                    self.r3 / config.meter_to_centimeter,
                    self.r3_list,
                    self.r2_r3_list,
                )

            Deformation.change_r2(
                self.points_for_3D_TX,
                self.points_for_3D_RX,
                self.y_move,
                self.r2 / config.meter_to_centimeter,
                self.r3 / config.meter_to_centimeter,
                self.r2_list,
                self.r2_r3_list,
            )

        elif changed_r3:
            Deformation.change_r3(
                self.points_for_3D_TX,
                self.points_for_3D_RX,
                self.y_move,
                self.r2 / config.meter_to_centimeter,
                self.r3 / config.meter_to_centimeter,
                self.r3_list,
                self.r2_r3_list,
            )
        elif changed_r4:
            Deformation.change_r4(
                self.points_for_3D_TX,
                self.points_for_3D_RX,
                self.y_move,
                self.r4 / config.meter_to_centimeter,
                self.r4_list,
            )

        # Common processing for all shape changes
        Deformation.change_X_test(self.X_test, self.points_for_3D_TX)
        self.values = Prediction.mu_star(
            self.X_train,
            self.X_test,
            self.K_inv_Y_train_mag,
            self.lengthscale,
            self.scale,
            self.mean_constant_mag,
        ).reshape(-1)
        self.dBvalues = Visualize.convert_to_db(self.values)
        self.efficiency, self.var_efficiency = Prediction.mu_star_efficiency(
            self.X_train_efficiency,
            self.X_test,
            self.K_inv_Y_train_efficiency,
            self.lengthscale_efficiency,
            self.scale_efficiency,
            self.K_inv_efficiency,
            self.mean_constant_efficiency,
            self.noise_efficiency,
        )

        # Mark vector cache as needing update
        self._vector_needs_update = True

    def get_values(self):
        """Get current electromagnetic field magnitude prediction values.

        Returns:
            Current predicted field magnitude values at all prediction points

        """
        return self.values

    def get_efficiency(self):
        """Get current power transfer efficiency prediction.

        Returns:
            Current predicted efficiency value with uncertainty bounds


        """
        return self.efficiency

    def get_prediction_points(self):
        """Get current prediction point coordinates after deformation.

        Returns:
            3D coordinates of prediction points after linear blend skinning deformation

        Note:
            for clarity about the coordinate system.

        """
        return self.points_for_3D_prediction_move

    def test_setup(self) -> bool:
        """Test visualization setup for pytest."""
        try:
            ps.init()
            self.visualization_prepare()
            ps.set_user_callback(self.callback)
            ps.shutdown()
            return True
        except Exception:
            return False

    def test_predictions(self) -> bool:
        """Test prediction pipeline for pytest."""
        try:
            self.prediction_prepare()
            return (
                hasattr(self, "efficiency")
                and self.efficiency is not None
                and hasattr(self, "values")
                and self.values is not None
                and hasattr(self, "magnetic_vector")
                and self.magnetic_vector is not None
            )
        except Exception:
            return False


if __name__ == "__main__":
    shield_number = int(
        input(
            f"Please input the shield number (from 1 to {config.n_shield_shape}): ",
        ),
    )
    if shield_number < 1 or shield_number > config.n_shield_shape:
        sys.exit(
            f"Please input the number from 1 to {config.n_shield_shape}",
        )
    visualize = Visualize()
    visualize.visualize(shield_number)
