"""Main CLI interface for EMFieldML electromagnetic field toolkit.

This module provides the main command-line interface for the
electromagnetic field machine learning toolkit.
"""

import click

from EMFieldML.cli.commands import demo


@click.group()
def cli() -> None:
    """EMFieldML - Electromagnetic Field Machine Learning Toolkit."""


cli.add_command(demo.demo)

if __name__ == "__main__":
    cli()
