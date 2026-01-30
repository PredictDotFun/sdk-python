"""Internal utility functions for the Predict SDK."""

from __future__ import annotations

import random
from decimal import ROUND_DOWN, Decimal
from typing import Any

from eth_abi import encode  # type: ignore[attr-defined]
from web3 import Web3

from predict_sdk.constants import MAX_SALT


def float_to_wei(value: float, precision: int) -> int:
    """
    Convert a floating-point value to wei using exact decimal arithmetic.

    Avoids IEEE 754 floating-point precision errors by converting
    the float to a string first, then using Python's Decimal module.

    Uses ROUND_DOWN to match Solidity's integer division behavior.

    Args:
        value: The floating-point value to convert (e.g., 0.46 for a price).
        precision: The precision multiplier (e.g., 10**18 for wei).

    Returns:
        The value converted to wei as an integer.

    Example:
        >>> float_to_wei(0.46, 10**18)
        460000000000000000
        >>> float_to_wei(0.421031, 10**18)
        421031000000000000  # Correct! (not 421030999999999936)
    """
    d = Decimal(str(value)) * Decimal(precision)
    return int(d.quantize(Decimal("1"), rounding=ROUND_DOWN))


def generate_order_salt() -> str:
    """
    Generate a random salt for an order.

    Returns:
        A random numeric string value for the salt.
    """
    return str(random.randint(0, MAX_SALT))


def retain_significant_digits(num: int, significant_digits: int) -> int:
    """
    Retain the specified number of significant digits.

    In the case of negative numbers, the significant digits are retained as
    expected without the sign affecting the calculation.

    Args:
        num: The integer number to truncate.
        significant_digits: The number of significant digits to retain.

    Returns:
        The integer number with the specified significant digits retained.
    """
    if num == 0:
        return 0

    is_negative = num < 0  # Check if the number is negative
    abs_num = -num if is_negative else num  # Work with the absolute value

    # Convert to string to find magnitude (length before trailing zeros)
    str_num = str(abs_num)
    magnitude = len(str_num)

    # Calculate divisor to remove excess digits
    excess = magnitude - significant_digits
    if excess <= 0:
        return num  # Return original number if no truncation is needed

    divisor: int = 10**excess

    # Divide then multiply to truncate, and restore the sign
    result: int = (abs_num // divisor) * divisor
    return -result if is_negative else result


def hash_kernel_message(message_hash: str) -> str:
    """
    Hash a message for Kernel smart wallet.

    Args:
        message_hash: The message hash to wrap (hex string with 0x prefix).

    Returns:
        The wrapped message hash as a hex string.
    """
    # "Kernel(bytes32 hash)" type hash
    kernel_type_hash = Web3.keccak(text="Kernel(bytes32 hash)")

    # Convert message_hash from hex string to bytes
    message_hash_bytes = (
        bytes.fromhex(message_hash[2:])
        if message_hash.startswith("0x")
        else bytes.fromhex(message_hash)
    )

    # Encode [bytes32, bytes32]
    encoded = encode(["bytes32", "bytes32"], [kernel_type_hash, message_hash_bytes])

    return "0x" + Web3.keccak(encoded).hex()


def eip712_wrap_hash(message_hash: str, domain: dict[str, Any]) -> str:
    """
    Wrap a message hash with EIP-712 domain separator.

    This is used for Predict account (Kernel smart wallet) signing.

    Args:
        message_hash: The message hash (hex string with 0x prefix).
        domain: The EIP-712 domain containing name, version, chainId, verifyingContract.

    Returns:
        The wrapped hash as a hex string.
    """
    # Calculate domain separator
    domain_separator = _hash_eip712_domain(domain)

    # Get the final message hash using Kernel wrapper
    final_message_hash = hash_kernel_message(message_hash)

    # Convert to bytes
    final_hash_bytes = (
        bytes.fromhex(final_message_hash[2:])
        if final_message_hash.startswith("0x")
        else bytes.fromhex(final_message_hash)
    )

    # Concatenate: 0x1901 + domainSeparator + messageHash
    data = b"\x19\x01" + domain_separator + final_hash_bytes

    return "0x" + Web3.keccak(data).hex()


def _hash_eip712_domain(domain: dict[str, Any]) -> bytes:
    """
    Hash an EIP-712 domain.

    Args:
        domain: The domain containing name, version, chainId, verifyingContract.

    Returns:
        The domain separator as bytes.
    """
    # EIP-712 Domain Type Hash
    domain_type = (
        "EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)"
    )
    domain_type_hash = Web3.keccak(text=domain_type)

    # Hash the name and version strings
    name_hash = Web3.keccak(text=domain["name"])
    version_hash = Web3.keccak(text=domain["version"])

    # Convert chainId to int if necessary
    chain_id = int(domain["chainId"])

    # Convert verifyingContract to checksum address if it's a string
    verifying_contract = domain["verifyingContract"]
    if isinstance(verifying_contract, str):
        verifying_contract = Web3.to_checksum_address(verifying_contract)

    # Encode the domain struct
    encoded = encode(
        ["bytes32", "bytes32", "bytes32", "uint256", "address"],
        [domain_type_hash, name_hash, version_hash, chain_id, verifying_contract],
    )

    return Web3.keccak(encoded)


def compute_order_hash(
    typed_data: dict[str, Any],
) -> str:
    """
    Compute the hash of EIP-712 typed data for an order.

    Args:
        typed_data: The EIP-712 typed data structure.

    Returns:
        The hash as a hex string.
    """
    from eth_account.messages import _hash_eip191_message, encode_typed_data

    encoded = encode_typed_data(full_message=typed_data)
    return "0x" + _hash_eip191_message(encoded).hex()
