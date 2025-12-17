"""Tests for utility functions."""

from __future__ import annotations

import pytest

from predict_sdk._internal.utils import (
    compute_order_hash,
    eip712_wrap_hash,
    generate_order_salt,
    hash_kernel_message,
    retain_significant_digits,
)
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


class TestHashKernelMessage:
    """Test hash_kernel_message function."""

    def test_returns_hash_with_0x_prefix(self):
        """hash_kernel_message should return hash with 0x prefix."""
        # Use a sample hash (with 0x prefix)
        message_hash = "0x" + "a" * 64
        result = hash_kernel_message(message_hash)

        assert result.startswith("0x"), f"Hash should start with '0x', got: {result}"
        assert len(result) == 66, f"Hash should be 66 chars, got: {len(result)}"
        assert all(c in "0123456789abcdef" for c in result[2:])

    def test_handles_hash_without_0x_prefix(self):
        """hash_kernel_message should handle hash without 0x prefix."""
        # Use a sample hash (without 0x prefix)
        message_hash = "a" * 64
        result = hash_kernel_message(message_hash)

        assert result.startswith("0x"), f"Hash should start with '0x', got: {result}"
        assert len(result) == 66


class TestEip712WrapHash:
    """Test eip712_wrap_hash function."""

    def test_returns_hash_with_0x_prefix(self):
        """eip712_wrap_hash should return hash with 0x prefix."""
        message_hash = "0x" + "b" * 64
        domain = {
            "name": "TestDomain",
            "version": "1",
            "chainId": 56,
            "verifyingContract": "0x" + "c" * 40,
        }

        result = eip712_wrap_hash(message_hash, domain)

        assert result.startswith("0x"), f"Hash should start with '0x', got: {result}"
        assert len(result) == 66, f"Hash should be 66 chars, got: {len(result)}"
        assert all(c in "0123456789abcdef" for c in result[2:])


class TestComputeOrderHash:
    """Test compute_order_hash function."""

    def test_compute_order_hash_matches_ts_sdk(self):
        """Hash should match the TS SDK's TypedDataEncoder.hash() output."""
        # Static typed data with known expected hash from TS SDK
        typed_data = {
            "types": {
                "EIP712Domain": [
                    {"name": "name", "type": "string"},
                    {"name": "version", "type": "string"},
                    {"name": "chainId", "type": "uint256"},
                    {"name": "verifyingContract", "type": "address"},
                ],
                "Order": [
                    {"name": "salt", "type": "uint256"},
                    {"name": "maker", "type": "address"},
                    {"name": "signer", "type": "address"},
                    {"name": "taker", "type": "address"},
                    {"name": "tokenId", "type": "uint256"},
                    {"name": "makerAmount", "type": "uint256"},
                    {"name": "takerAmount", "type": "uint256"},
                    {"name": "expiration", "type": "uint256"},
                    {"name": "nonce", "type": "uint256"},
                    {"name": "feeRateBps", "type": "uint256"},
                    {"name": "side", "type": "uint8"},
                    {"name": "signatureType", "type": "uint8"},
                ],
            },
            "primaryType": "Order",
            "domain": {
                "name": "predict.fun CTF Exchange",
                "version": "1",
                "chainId": 56,
                "verifyingContract": "0x8BC070BEdAB741406F4B1Eb65A72bee27894B689",
            },
            "message": {
                "salt": "123456789",
                "maker": "0x1234567890123456789012345678901234567890",
                "signer": "0x1234567890123456789012345678901234567890",
                "taker": "0x0000000000000000000000000000000000000000",
                "tokenId": "12345",
                "makerAmount": "1000000000000000000",
                "takerAmount": "2000000000000000000",
                "expiration": "4102444800",
                "nonce": "0",
                "feeRateBps": "100",
                "side": 0,
                "signatureType": 0,
            },
        }

        result = compute_order_hash(typed_data)

        # Expected hash from TS SDK's TypedDataEncoder.hash()
        expected_hash = "0x814000c89efa61ae42a2bcc4c98e06e90c11480b95a12edea00e3411ec76821d"
        assert result == expected_hash, f"Hash mismatch: got {result}, expected {expected_hash}"

    def test_returns_hash_with_0x_prefix(self):
        """compute_order_hash should return hash with 0x prefix."""
        # Create a minimal valid EIP-712 typed data structure
        typed_data = {
            "types": {
                "EIP712Domain": [
                    {"name": "name", "type": "string"},
                    {"name": "version", "type": "string"},
                    {"name": "chainId", "type": "uint256"},
                    {"name": "verifyingContract", "type": "address"},
                ],
                "Order": [
                    {"name": "salt", "type": "uint256"},
                    {"name": "maker", "type": "address"},
                ],
            },
            "primaryType": "Order",
            "domain": {
                "name": "TestExchange",
                "version": "1",
                "chainId": 56,
                "verifyingContract": "0x" + "d" * 40,
            },
            "message": {
                "salt": 123456789,
                "maker": "0x" + "e" * 40,
            },
        }

        result = compute_order_hash(typed_data)

        assert result.startswith("0x"), f"Hash should start with '0x', got: {result}"
        assert len(result) == 66, f"Hash should be 66 chars, got: {len(result)}"
        assert all(c in "0123456789abcdef" for c in result[2:])
