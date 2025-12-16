import math


def normal_cdf(x):
    """Standard normal CDF using error function."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def two_proportion_z_test(p1, p2, n1, n2=None):
    """
    Two-proportion z-test for significance.

    Args:
        p1: Proportion for group 1 (e.g., 0.4383)
        p2: Proportion for group 2 (e.g., 0.4617)
        n1: Sample size for group 1
        n2: Sample size for group 2 (defaults to n1 if not provided)

    Returns:
        z-statistic, p-value (two-tailed)
    """
    if n2 is None:
        n2 = n1

    # Pooled proportion
    pooled = (p1 * n1 + p2 * n2) / (n1 + n2)

    # Standard error
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))

    # Z-statistic
    z = (p1 - p2) / se

    # Two-tailed p-value using normal CDF approximation
    p_value = 2 * (1 - normal_cdf(abs(z)))

    return z, p_value


# two_proportion_z_test(0.2464, 0.3043, 69)
two_proportion_z_test(0.405, 0.5667, 600)
