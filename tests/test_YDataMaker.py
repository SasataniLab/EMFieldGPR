from EMFieldML.Learning.YDataMaker import TargetDataBuilder


def test_calculate_magnitude():
    """Test magnitude calculation

    Input: magnetic field components (Hx, Hx_theta, Hy, Hy_theta, Hz, Hz_theta)
    Output: magnitude value (float)
    """
    # Test with sample field data
    Hx, Hx_theta = 1.0, 0.0
    Hy, Hy_theta = 2.0, 90.0
    Hz, Hz_theta = 3.0, 180.0

    result = TargetDataBuilder.calculate_magnitude(
        Hx, Hx_theta, Hy, Hy_theta, Hz, Hz_theta
    )

    # Check that result is a positive number
    assert isinstance(result, float)
    assert result > 0

    # Test with known values for validation (this function does complex angle optimization)
    # The result should be positive and reasonable for the given inputs
    assert result > 0, f"Magnitude should be positive, got {result}"
    assert result < 10.0, f"Magnitude seems too large: {result}"

    # Test edge cases (avoiding division by zero)
    single_result = TargetDataBuilder.calculate_magnitude(1, 0, 0, 0, 0, 0)
    assert (
        single_result > 0
    ), f"Single component should give positive result, got {single_result}"


def test_calculate_vector():
    """Test vector calculation

    Input: magnetic field components (Hx, Hx_theta, Hy, Hy_theta, Hz, Hz_theta)
    Output: theta and phi angles (float, float)
    """
    Hx, Hx_theta = 1.0, 0.0
    Hy, Hy_theta = 2.0, 90.0
    Hz, Hz_theta = 3.0, 180.0

    theta, phi = TargetDataBuilder.calculate_vector(
        Hx, Hx_theta, Hy, Hy_theta, Hz, Hz_theta
    )

    # Check that results are numbers
    assert isinstance(theta, float)
    assert isinstance(phi, float)


def test_calculateTrilinearWeight():
    """Test trilinear weight calculation

    Input: prediction point, current point, base points A and B
    Output: interpolation weight (float between 0 and 1)
    """
    prePoint = [1.0, 2.0, 3.0]
    nowPoint = [1.5, 2.5, 3.5]
    basedPointA = [1.0, 2.0, 3.0]
    basedPointB = [2.0, 3.0, 4.0]

    result = TargetDataBuilder.calculateTrilinearWeight(
        prePoint, nowPoint, basedPointA, basedPointB
    )

    # Check that result is a number between 0 and 1
    assert isinstance(result, float)
    assert 0 <= result <= 1

    # Test edge cases with known results
    # When prediction point equals base point A, weight should be 1
    assert (
        TargetDataBuilder.calculateTrilinearWeight(
            prePoint, prePoint, basedPointA, basedPointB
        )
        == 1.0
    )

    # When prediction point equals base point B, weight should be 0
    assert (
        TargetDataBuilder.calculateTrilinearWeight(
            prePoint, basedPointB, basedPointA, basedPointB
        )
        == 0.0
    )
