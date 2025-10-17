"""Configuration management for EMFieldML electromagnetic field toolkit.

This module provides a clean, organized configuration system with separate
modules for different types of configuration data.
"""

from .logger import get_logger
from .paths import EMFieldMLPaths
from .settings import EMFieldMLConfig
from .templates import EMFieldMLTemplate

# Create singleton instances
config = EMFieldMLConfig()
paths = EMFieldMLPaths()
template = EMFieldMLTemplate()

__all__ = ["config", "paths", "template", "get_logger"]
