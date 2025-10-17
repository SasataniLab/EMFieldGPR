"""
Helper class to run FEKO simulations.
"""

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import skrf as rf
from dotenv import load_dotenv

from EMFieldML.config import config, paths, template

load_dotenv()
cadfeko_path = os.getenv("CADFEKO_PATH")


class FekoRunner:
    """Class for running FEKO simulations and managing STL files.

    This class provides methods to run FEKO simulations using Lua scripts,
    manage input and output files, and handle the generation of STL files.

    To fully automate a simulation in FEKO, you'd implement a workflow
    that adjusts for model complexity. The process can be summarized as:

    1. Model Setup: Define the parameters for the shield and coils to
    create the initial model.

    2. Initial Inductance Calculation: Run a simulation to calculate
    the inductance of the model. This value is crucial for determining
    the starting point for resistance.

    3. Optimal Resistance Simulation: Use the calculated inductance to
    set an initial optimal resistance value. A second simulation is
    then performed with this value.

    4. Iterative Refinement (for complex models): If the initial optimal
    resistance fails to produce a stable or accurate result
    (common with complex models), the resistance value is iteratively adjusted.
    This fine-tuning process, typically completed in a few steps (up to five),
    ensures the simulation converges properly.

    5. Final Magnetic Field Simulation: With the final, verified component
    values, the magnetic field is simulated to produce the desired results.

    """

    @staticmethod
    def run_feko(
        lua_script_path: Path,
        input_data: Path,
        num_trial: int = config.lua_trial,
    ) -> None:
        """
        Run FEKO by lua file.

        Args:
            lua_script_path (str): The path of the lua file.
            input_data (str): The input data for the lua file.
            num_trial (int): The number of trials to run the lua file.

        Returns:
            no return

        """
        # subprocess.runを使用してLuaスクリプトを実行
        # Note: S603 warning - executing FEKO CAD software with controlled parameters
        for _i in range(num_trial):
            process = subprocess.Popen(  # noqa: S603
                [cadfeko_path, r"--non-interactive", r"--run-script", lua_script_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
            )

            process.stdin.write(input_data)
            process.stdin.close()
            return_code = process.wait()
            if return_code == 0:
                break
            print(f"Script failed with return code {return_code}. Retrying...")

    @staticmethod
    def make_coil_stl(
        num_trial: int = config.lua_trial,
    ) -> None:
        """
        Run a lua file "coil_stl.lua" to create STL file.

        Args:
            num_trial (int): The number of trials to run the lua file.

        Returns:
            no return

        Note:
            This function corresponds to Supplementary Figure 1a in the paper.

        """
        lua_script_path = paths.LUA_DIR / "coil_stl.lua"
        save_stl_file_name = paths.CIRCLE_STL_DIR / "coil.stl"

        input_data = f"{save_stl_file_name}"

        FekoRunner.run_feko(lua_script_path, input_data, num_trial)

    @staticmethod
    def make_stl(
        n_shield_shape: int,
        num_trial: int = config.lua_trial,
        path_circle_dir: Path = paths.CIRCLE_DIR,
        path_save_dir: Path = paths.CIRCLE_STL_DIR,
        path_circle_file: str = template.circle_param,
        path_save_file: str = template.circle_stl,
    ) -> None:
        """
        Run a lua file "make_stl.lua" to create STL file.

        Creating an STL file supports performing shape modeling of an Aligned-edge Polycube mesh.

        Args:
            n_shield_shape: Number of shield shapes to process
            num_trial: Number of trials for FEKO execution
            path_circle_dir: Directory containing circle parameter files
            path_save_dir: Directory to save STL files
            path_circle_file: Template for circle parameter file names
            path_save_file: Template for STL file names

        Note:
            This function corresponds to Figure 4 in the paper.

        """
        n_index = n_shield_shape + 1  # Index in Natural number

        print(f"start No.{format(n_index)}")
        lua_script_path = paths.LUA_DIR / "make_stl.lua"
        load_pattern_file_name = path_circle_dir / path_circle_file.format(
            index=n_index
        )
        save_stl_file_name = path_save_dir / path_save_file.format(index=n_index)

        input_data = f"{load_pattern_file_name}\n{save_stl_file_name}"

        FekoRunner.run_feko(lua_script_path, input_data, num_trial)

    @staticmethod
    def make_stl_list(
        n_shield_shape: int = config.n_shield_shape,
    ) -> None:
        """
        Run a lua file in FEKO to create STL files (372 types) from ferrite shield parameters.
        """
        for i in range(n_shield_shape):
            FekoRunner.make_stl(i)

    @staticmethod
    def make_stl_test_list(
        n_shield_shape: int = config.n_test_data,
    ) -> None:
        """
        Run a lua file in FEKO to create STL files (372 types) from ferrite shield parameters.
        """
        for i in range(1, n_shield_shape + 1):
            FekoRunner.make_stl(
                i,
                path_circle_dir=paths.TEST_CIRCLE_MOVE_X_DIR,
                path_save_dir=paths.TEST_CIRCLE_STL_DIR,
                path_circle_file=template.test_circle_move,
                path_save_file=template.test_circle_stl,
            )

    @staticmethod
    def check_validity(
        input_path_check: Path,
        efficiency: float,
    ) -> int:
        """Check validity of FEKO simulation results.

        This function checks the validity of the FEKO simulation results by comparing
        the real efficiency with the expected efficiency and the phase of the voltage
        source.

        Args:
            input_path_check: Path to the FEKO output file to check
            efficiency: Expected efficiency value for comparison

        Returns:
            0 if the results are valid, 1 if they are invalid

        """
        with Path(input_path_check).open() as f:
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

        lines_strip = [line.strip() for line in lines]
        list_rownum = [
            i
            for i, line_s in enumerate(lines_strip)
            if "DATA OF THE VOLTAGE SOURCE NO. 1" in line_s
        ]
        line_split_efficiency = lines_strip[list_rownum[0] + 5].split()
        phase = float(line_split_efficiency[6])
        if abs(phase) > 5.0 and abs(efficiency - real_efficiency) / efficiency > 0.01:
            return 1
        return 0

    @staticmethod
    def run_solver(
        number: int,
        path_save_dir: Path = paths.CIRCLE_FEKO_RAW_DIR,
        path_circle_dir: Path = paths.CIRCLE_DIR,
        path_circle_file: str = template.circle_move_param,
        num_trial: int = config.lua_trial,
    ) -> None:
        """
        Run a simulation of ferrite shield.
        Firstly, run simulation_s2p.lua to calculate Sparameter.
        Secondly, calculate some parameter for resonance.

        Args:
            number: Shield number to process
            path_save_dir: Directory to save FEKO raw data
            path_circle_dir: Directory containing circle parameter files
            path_circle_file: Template for circle parameter file names
            num_trial: Number of trials for FEKO execution

        """
        print(f"start No.{number}")
        lua_script_path = paths.LUA_DIR / "simulation_s2p.lua"

        save_file_name = path_save_dir / f"Circle_{number}_2"
        load_file_name = path_circle_dir / path_circle_file.format(index=number)

        input_data = f"{save_file_name}\n{load_file_name}"
        # subprocess.runを使用してLuaスクリプトを実行。

        FekoRunner.run_feko(lua_script_path, input_data, num_trial)

        # s2pファイルを読み込む
        s2p_filename = path_save_dir / template.raw_data_sparameter.format(index=number)
        ntwk = rf.Network(s2p_filename)
        z_params = ntwk.z[0]
        omega = 2 * np.pi * 85 * 10**3
        r11 = z_params[0][0].real
        r22 = z_params[1][1].real
        rm = z_params[0][1].real
        x22 = z_params[1][1].imag
        xm = z_params[0][1].imag
        R = r11 * r22 - rm**2
        kQ2 = (rm**2 + xm**2) / R
        RL = R * np.sqrt(1 + kQ2) / r11
        xopt = (rm * xm - r11 * x22) / r11
        C = -1 / (omega * xopt)
        Lm = -1 * xm / omega
        efficiency = kQ2 / (1 + (1 + kQ2) ** (1 / 2)) ** 2
        print(f"RL = {RL}, Lm = {Lm}, C = {C}")
        V1 = (omega * Lm + r11 * (r22 + RL) / (omega * Lm)) * (2 / RL) ** (1 / 2)

        lua_script_path = paths.LUA_DIR / "simulation_field.lua"
        save_file_name = path_save_dir / f"Circle_{number}_3"
        input_data = f"{save_file_name}\n{load_file_name}\n{C}\n{V1}\n{RL}"

        FekoRunner.run_feko(lua_script_path, input_data, num_trial)

        input_path_check = path_save_dir / template.raw_data.format(index=number)
        validity = FekoRunner.check_validity(input_path_check, efficiency)
        if validity == 1:
            lua_script_path = paths.LUA_DIR / "simulation_optimization.lua"

            save_file_name = path_save_dir / "circle_optimum" / f"Circle_{number}"
            input_data = f"{save_file_name}\n{load_file_name}\n{C}\n{V1}\n{RL}"

            print("Optimization.")
            FekoRunner.run_feko(lua_script_path, input_data)

            input_path_optimization = (
                path_save_dir
                / "circle_optimum"
                / template.raw_data_optimization.format(index=number)
            )
            with Path(input_path_optimization).open() as f:
                lines = f.readlines()
            lines_strip = [line.strip() for line in lines]
            list_rownum = [
                i
                for i, line_s in enumerate(lines_strip)
                if "RESULTS FOR LOADS" in line_s
            ]
            line_split_efficiency = lines_strip[list_rownum[0] + 5].split()
            impedance = float(line_split_efficiency[9])
            C = 1 / omega / abs(impedance)

            lua_script_path = paths.LUA_DIR / "simulation_field.lua"
            save_file_name = path_save_dir / f"Circle_{number}_3"
            input_data = f"{save_file_name}\n{load_file_name}\n{C}\n{V1}\n{RL}"
            print("Simulation for Out file after Optimization.")
            FekoRunner.run_feko(lua_script_path, input_data)

    @staticmethod
    def delete_warning(
        number: int,
        path_save_dir: Path = paths.CIRCLE_FEKO_RAW_DIR,
    ) -> None:
        """
        Delete error sentences.

        You use this to eliminate the errors that appear in the log
        during a FEKO simulation, which happen in areas where the calculations are difficult.

        Args:
            number: File number to process
            path_save_dir: Directory containing files to clean

        """
        filename = path_save_dir / f"Circle_{number}_3.out"

        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            try:
                # Open the input file and copy the content to the temporary file, skipping lines that start with 'WARNING'
                with (
                    Path(filename).open() as infile,
                    Path(tmp.name).open("w") as outfile,
                ):
                    for line in infile:
                        if (not line.lstrip().startswith("WARNING")) and (
                            not line.lstrip().startswith("Received")
                        ):
                            outfile.write(line)
            finally:
                # Close the temporary file
                tmp.close()

        # Remove the original file
        Path(filename).unlink()

        # Move the new file
        Path(tmp.name).rename(filename)

    @staticmethod
    def run_simulation_list(
        x_train_list: list[int],
    ) -> None:
        """
        Run a simulation of ferrite shield.

        Args:
            x_train_list : The list of the number used train data.

        Returns:
            no return

        """
        for i in x_train_list:
            FekoRunner.run_solver(i)
            FekoRunner.delete_warning(i)

    @staticmethod
    def run_simulation_test_list(
        x_train_list: list[int],
    ) -> None:
        """
        Run a simulation of ferrite shield.

        Args:
            x_train_list : The list of the number used train data.

        Returns:
            no return

        """
        for i in x_train_list:
            FekoRunner.run_solver(
                i,
                path_save_dir=paths.TEST_CIRCLE_FEKO_RAW_DIR,
                path_circle_dir=paths.TEST_CIRCLE_MOVE,
                path_circle_file=template.test_circle_move_param,
            )
            FekoRunner.delete_warning(i, path_save_dir=paths.TEST_CIRCLE_FEKO_RAW_DIR)
