"""Float comparison utilities for electromagnetic field validation.

This module provides a class for comparing lists of floats within a specified precision.
"""

from pathlib import Path

import numpy as np


class FloatAssert:
    """Float comparison utility for electromagnetic field validation."""

    def __init__(self, precision: float = 1e-7):
        """
        Initialize the validator with a specified precision.

        :param precision: The precision threshold for float comparisons.
        """
        self.precision = precision

    def read_floats_from_file(self, file_path: Path) -> np.ndarray:
        """
        Read a list of floats from a specified file, assuming space-separated values in each line.

        :param file_path: Path to the file to read.
        :return: A NumPy array of floats read from the file.
        """
        return np.loadtxt(file_path)

    def compare_files(self, file1: Path, file2: Path) -> bool:
        """
        Compare two files containing lists of floats.

        :param file1: Path to the first file.
        :param file2: Path to the second file.
        :return: True if the files are equivalent within the specified precision, False otherwise.
        """
        array1 = self.read_floats_from_file(file1)
        array2 = self.read_floats_from_file(file2)

        return self.compare_float_arrays(array1, array2)

    def compare_float_arrays(self, array1: np.ndarray, array2: np.ndarray) -> bool:
        """
        Compare two arrays of floats within the specified precision.

        :param array1: The first array of floats.
        :param array2: The second array of floats.
        :return: True if the arrays are equivalent within the precision, False otherwise.
        """
        if array1.shape != array2.shape:
            print("Arrays are of different shapes.")
            return False

        differences = np.abs(array1 - array2) > self.precision
        if np.any(differences):
            indices = np.where(differences)
            for index in zip(*indices, strict=True):  #
                print(
                    f"Difference found at index {index}: {array1[index]} (file1) vs {array2[index]} (file2)"
                )
            return False

        return True
