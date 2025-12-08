"""Pytest fixtures for the Predict SDK tests."""

from __future__ import annotations

import pytest
from eth_account import Account

from predict_sdk import ChainId, OrderBuilder
from predict_sdk.constants import ADDRESSES_BY_CHAIN_ID
from predict_sdk.logger import Logger

# A deterministic test private key (DO NOT USE IN PRODUCTION)
TEST_PRIVATE_KEY = "0x" + "a" * 64


@pytest.fixture
def builder() -> OrderBuilder:
    """Create an OrderBuilder instance without a signer (for read-only operations)."""
    return OrderBuilder.make(ChainId.BNB_MAINNET)


@pytest.fixture
def builder_with_signer() -> OrderBuilder:
    """Create an OrderBuilder instance with a signer for order building tests."""
    signer = Account.from_key(TEST_PRIVATE_KEY)
    # Use direct construction to avoid RPC calls
    # precision is the exponent (18), __init__ does 10**precision internally
    return OrderBuilder(
        chain_id=ChainId.BNB_MAINNET,
        precision=18,
        addresses=ADDRESSES_BY_CHAIN_ID[ChainId.BNB_MAINNET],
        generate_salt_fn=lambda: "123456789",
        logger=Logger("ERROR"),
        signer=signer,
    )


@pytest.fixture
def builder_testnet() -> OrderBuilder:
    """Create an OrderBuilder instance for testnet without a signer."""
    return OrderBuilder.make(ChainId.BNB_TESTNET)


@pytest.fixture
def mock_private_key() -> str:
    """A mock private key for testing (DO NOT USE IN PRODUCTION)."""
    return TEST_PRIVATE_KEY


@pytest.fixture
def mock_orderbook() -> dict:
    """A mock orderbook for testing."""
    from predict_sdk import Book

    return Book(
        market_id=1,
        update_timestamp_ms=int(__import__("time").time() * 1000),
        asks=[
            (0.50, 100.0),
            (0.51, 200.0),
            (0.52, 300.0),
        ],
        bids=[
            (0.49, 100.0),
            (0.48, 200.0),
            (0.47, 300.0),
        ],
    )
