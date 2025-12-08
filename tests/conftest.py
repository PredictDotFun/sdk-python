"""Pytest fixtures for the Predict SDK tests."""

from __future__ import annotations

import pytest

from predict_sdk import ChainId, OrderBuilder


@pytest.fixture
def builder() -> OrderBuilder:
    """Create an OrderBuilder instance without a signer."""
    return OrderBuilder.make(ChainId.BNB_MAINNET)


@pytest.fixture
def builder_testnet() -> OrderBuilder:
    """Create an OrderBuilder instance for testnet without a signer."""
    return OrderBuilder.make(ChainId.BNB_TESTNET)


@pytest.fixture
def mock_private_key() -> str:
    """A mock private key for testing (DO NOT USE IN PRODUCTION)."""
    return "0x" + "a" * 64


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
