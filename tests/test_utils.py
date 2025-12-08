"""Tests for utility functions."""

from __future__ import annotations

import pytest

from predict_sdk._internal.utils import generate_order_salt, retain_significant_digits
from predict_sdk.constants import MAX_SALT


class TestRetainSignificantDigits:
    """Test significant digit retention."""

    @pytest.mark.parametrize(
        "num,digits,expected",
        [
            (123456789, 3, 123000000),
            (123456789, 5, 123450000),
            (100000000, 3, 100000000),
            (0, 5, 0),
            (-123456789, 3, -123000000),
            (999999999, 3, 999000000),
            (100, 5, 100),  # No truncation needed
            (12345, 5, 12345),  # Exact match
            (1, 3, 1),  # Single digit
        ],
    )
    def test_retain_digits(self, num: int, digits: int, expected: int):
        """Test retainSignificantDigits with various inputs."""
        assert retain_significant_digits(num, digits) == expected

    def test_never_increases_value(self):
        """Retained value should never be greater than original."""
        test_values = [123456789, 987654321, 100000000, 999999999]
        for num in test_values:
            for digits in range(1, 10):
                result = retain_significant_digits(num, digits)
                assert abs(result) <= abs(num)


class TestGenerateOrderSalt:
    """Test salt generation."""

    def test_generates_string(self):
        """Salt should be a string."""
        salt = generate_order_salt()
        assert isinstance(salt, str)

    def test_generates_numeric_string(self):
        """Salt should be a numeric string."""
        salt = generate_order_salt()
        assert salt.isdigit() or (salt.startswith("-") and salt[1:].isdigit())

    def test_within_max_salt(self):
        """Salt should be within MAX_SALT bounds."""
        for _ in range(100):
            salt = int(generate_order_salt())
            assert 0 <= salt <= MAX_SALT

    def test_generates_different_values(self):
        """Multiple calls should generate different values (with high probability)."""
        salts = [generate_order_salt() for _ in range(10)]
        # At least some should be different
        assert len(set(salts)) > 1
