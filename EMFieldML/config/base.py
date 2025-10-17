"""Base configuration classes for EMFieldML electromagnetic field toolkit."""

from typing import Any, Dict


class BaseConfig:
    """Base configuration class for EMFieldML electromagnetic field toolkit."""

    def dump_attrs(self) -> Dict[str, Any]:
        """Dump all attributes of the class.

        Returns:
            Dictionary containing all non-private attributes.

        """
        return {k: v for k, v in self.__dict__.items() if not k.startswith("__")}

    def show_attrs(self) -> None:
        """Show all attributes of the class."""
        for k, v in self.dump_attrs().items():
            print(f"{k}: {v}")
