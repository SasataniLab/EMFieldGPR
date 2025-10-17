"""Ferrite shield modeling and parameter generation.

Create parameters for the ferrite shield.
372 types of ferrite shield shapes are created.
By preparing 169 positions for these, a total of 62,868 pieces of data are created.


"""

from pathlib import Path

from EMFieldML.config import config, get_logger, paths, template

logger = get_logger(__name__)


class ShieldModeler:
    """Generate ferrite shield shapes and parameters for electromagnetic field modeling."""

    @staticmethod
    def _make_shield_shape(
        output_dir: Path,
    ) -> None:
        """
        Make ferrite shield shape data.

        Parameters
        ----------
        output_dir : Path
            Output directory
        config : EMFieldMLConfig
            Configuration of the ferrite shield

        Returns
        -------
        no return

        """
        count = 1
        for r1 in range(config.r1_min, config.r1_max, config.radius_step):
            for r3 in range(config.r3_min, r1, config.radius_step):
                for r2 in range(r3 + 1, r1 + 1, config.radius_step):
                    for r4 in config.r4_list:
                        for h1 in range(config.height_min, config.height_max):
                            for h2 in range(config.height_min, config.height_max):
                                filepath = output_dir / template.circle_param.format(
                                    index=count
                                )
                                with Path(filepath).open("w") as f:
                                    values = [r1, r2, r3, r4, h1, h2, h2]
                                    scaled_values = [
                                        value * config.scaler for value in values
                                    ]
                                    print(*scaled_values, file=f)
                                count += 1
                                logger.info(f"Saved {filepath}")

        # フェライトシールドが一段構成でコイルの下に一枚しかない場合のフェライト形状を作成
        for r1 in range(config.r1_min, config.r1_max, config.radius_step):
            for h1 in range(config.height_min, config.height_max):
                filepath = output_dir / template.circle_param.format(index=count)
                with Path(filepath).open("w") as f:
                    values = [r1, 13, 14, 4, h1, 0, 0]
                    scaled_values = [value * config.scaler for value in values]
                    print(*scaled_values, file=f)
                count += 1
                logger.info(f"Saved {filepath}")

    @staticmethod
    def _make_moved_shield_coordinate(
        output_dir: Path,
    ) -> None:
        """
        Add position data to the ferrite shape data created so far.
        This creates y_max * z_max data for one shape.

        Parameters
        ----------
        output_dir : Path
            Output directory
        config : EMFieldMLConfig
            Configuration of the ferrite shield

        """
        for shield_index in range(1, config.n_shield_shape + 1):
            filepath = output_dir / template.circle_param.format(index=shield_index)
            with Path(filepath).open() as f:
                data = f.readlines()
                num_list = list(map(float, data[0].split()))

            for y in range(config.y_grid):
                for z in range(config.z_grid):
                    y_move = y * config.y_step
                    z_move = z * config.z_step
                    filepath = output_dir / template.circle_move_param.format(
                        index=shield_index
                        + y * config.n_shield_shape * config.y_grid
                        + z * config.n_shield_shape
                    )
                    with Path(filepath).open("w") as f:
                        print(
                            num_list[0],
                            num_list[1],
                            num_list[2],
                            num_list[3],
                            num_list[4],
                            num_list[5],
                            num_list[6],
                            y_move,
                            z_move,
                            file=f,
                        )
                    logger.info(f"Saved {filepath}")

    @staticmethod
    def make_shield_coordinate() -> None:
        """
        Create parameters for the ferrite shield.
        """
        output_dir = paths.CIRCLE_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        ShieldModeler._make_shield_shape(
            output_dir=output_dir,
        )

        ShieldModeler._make_moved_shield_coordinate(
            output_dir=output_dir,
        )
