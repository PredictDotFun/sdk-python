"""Contract factory utilities for the Predict SDK."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from eth_account.signers.local import LocalAccount
from web3 import Web3
from web3.contract import Contract

from predict_sdk.abis import (
    CONDITIONAL_TOKENS_ABI,
    CTF_EXCHANGE_ABI,
    ECDSA_VALIDATOR_ABI,
    ERC20_ABI,
    KERNEL_ABI,
    NEG_RISK_ADAPTER_ABI,
    NEG_RISK_CTF_EXCHANGE_ABI,
    YIELD_BEARING_CONDITIONAL_TOKENS_ABI,
)
from predict_sdk.constants import Addresses


@dataclass
class Contracts:
    """Container for all contract instances."""

    # Exchange contracts
    ctf_exchange: Contract
    neg_risk_ctf_exchange: Contract
    yield_bearing_ctf_exchange: Contract
    yield_bearing_neg_risk_ctf_exchange: Contract

    # Conditional token contracts
    conditional_tokens: Contract
    neg_risk_conditional_tokens: Contract
    yield_bearing_conditional_tokens: Contract
    yield_bearing_neg_risk_conditional_tokens: Contract

    # Adapter contracts
    neg_risk_adapter: Contract
    yield_bearing_neg_risk_adapter: Contract

    # Token contracts
    usdt: Contract

    # Account contracts
    kernel: Contract
    ecdsa_validator: Contract


def make_contract(
    web3: Web3,
    address: str,
    abi: list[dict[str, Any]],
) -> Contract:
    """
    Create a contract instance.

    Args:
        web3: The Web3 instance.
        address: The contract address.
        abi: The contract ABI.

    Returns:
        A Web3 Contract instance.
    """
    return web3.eth.contract(
        address=Web3.to_checksum_address(address),
        abi=abi,
    )


def make_contracts(
    web3: Web3,
    addresses: Addresses,
    signer: LocalAccount | None = None,
) -> Contracts:
    """
    Create all contract instances.

    Args:
        web3: The Web3 instance.
        addresses: The contract addresses.
        signer: Optional signer for transactions.

    Returns:
        A Contracts container with all contract instances.
    """
    # Set default account if signer provided
    if signer:
        web3.eth.default_account = signer.address

    return Contracts(
        # Exchange contracts
        ctf_exchange=make_contract(web3, addresses.CTF_EXCHANGE, CTF_EXCHANGE_ABI),
        neg_risk_ctf_exchange=make_contract(
            web3, addresses.NEG_RISK_CTF_EXCHANGE, NEG_RISK_CTF_EXCHANGE_ABI
        ),
        yield_bearing_ctf_exchange=make_contract(
            web3, addresses.YIELD_BEARING_CTF_EXCHANGE, CTF_EXCHANGE_ABI
        ),
        yield_bearing_neg_risk_ctf_exchange=make_contract(
            web3, addresses.YIELD_BEARING_NEG_RISK_CTF_EXCHANGE, NEG_RISK_CTF_EXCHANGE_ABI
        ),
        # Conditional token contracts
        conditional_tokens=make_contract(
            web3, addresses.CONDITIONAL_TOKENS, CONDITIONAL_TOKENS_ABI
        ),
        neg_risk_conditional_tokens=make_contract(
            web3, addresses.NEG_RISK_CONDITIONAL_TOKENS, CONDITIONAL_TOKENS_ABI
        ),
        yield_bearing_conditional_tokens=make_contract(
            web3, addresses.YIELD_BEARING_CONDITIONAL_TOKENS, YIELD_BEARING_CONDITIONAL_TOKENS_ABI
        ),
        yield_bearing_neg_risk_conditional_tokens=make_contract(
            web3,
            addresses.YIELD_BEARING_NEG_RISK_CONDITIONAL_TOKENS,
            YIELD_BEARING_CONDITIONAL_TOKENS_ABI,
        ),
        # Adapter contracts
        neg_risk_adapter=make_contract(web3, addresses.NEG_RISK_ADAPTER, NEG_RISK_ADAPTER_ABI),
        yield_bearing_neg_risk_adapter=make_contract(
            web3, addresses.YIELD_BEARING_NEG_RISK_ADAPTER, NEG_RISK_ADAPTER_ABI
        ),
        # Token contracts
        usdt=make_contract(web3, addresses.USDT, ERC20_ABI),
        # Account contracts
        kernel=make_contract(web3, addresses.KERNEL, KERNEL_ABI),
        ecdsa_validator=make_contract(web3, addresses.ECDSA_VALIDATOR, ECDSA_VALIDATOR_ABI),
    )


def get_exchange_contract(
    contracts: Contracts,
    *,
    is_neg_risk: bool,
    is_yield_bearing: bool,
) -> Contract:
    """
    Get the appropriate exchange contract based on market type.

    Args:
        contracts: The contracts container.
        is_neg_risk: Whether this is a NegRisk market.
        is_yield_bearing: Whether this is a yield-bearing market.

    Returns:
        The appropriate exchange Contract instance.
    """
    if is_yield_bearing:
        if is_neg_risk:
            return contracts.yield_bearing_neg_risk_ctf_exchange
        return contracts.yield_bearing_ctf_exchange
    else:
        if is_neg_risk:
            return contracts.neg_risk_ctf_exchange
        return contracts.ctf_exchange


def get_conditional_tokens_contract(
    contracts: Contracts,
    *,
    is_neg_risk: bool,
    is_yield_bearing: bool,
) -> Contract:
    """
    Get the appropriate conditional tokens contract based on market type.

    Args:
        contracts: The contracts container.
        is_neg_risk: Whether this is a NegRisk market.
        is_yield_bearing: Whether this is a yield-bearing market.

    Returns:
        The appropriate ConditionalTokens Contract instance.
    """
    if is_yield_bearing:
        if is_neg_risk:
            return contracts.yield_bearing_neg_risk_conditional_tokens
        return contracts.yield_bearing_conditional_tokens
    else:
        if is_neg_risk:
            return contracts.neg_risk_conditional_tokens
        return contracts.conditional_tokens


def get_neg_risk_adapter_contract(
    contracts: Contracts,
    *,
    is_yield_bearing: bool,
) -> Contract:
    """
    Get the appropriate NegRisk adapter contract.

    Args:
        contracts: The contracts container.
        is_yield_bearing: Whether this is a yield-bearing market.

    Returns:
        The appropriate NegRiskAdapter Contract instance.
    """
    if is_yield_bearing:
        return contracts.yield_bearing_neg_risk_adapter
    return contracts.neg_risk_adapter
