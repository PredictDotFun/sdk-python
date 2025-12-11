"""Tests for order amount calculations."""

from __future__ import annotations

import time

import pytest

from predict_sdk import (
    Book,
    ChainId,
    InvalidQuantityError,
    LimitHelperInput,
    MarketHelperInput,
    MarketHelperValueInput,
    OrderBuilder,
    Side,
)


@pytest.fixture
def builder() -> OrderBuilder:
    """Create an OrderBuilder instance."""
    return OrderBuilder.make(ChainId.BNB_MAINNET)


@pytest.fixture
def fresh_orderbook() -> Book:
    """Create a fresh orderbook for testing."""
    return Book(
        market_id=1,
        update_timestamp_ms=int(time.time() * 1000),
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


class TestLimitOrderAmounts:
    """Test limit order amount calculations."""

    def test_buy_order_basic(self, builder: OrderBuilder):
        """Test basic BUY order calculation."""
        amounts = builder.get_limit_order_amounts(
            LimitHelperInput(
                side=Side.BUY,
                price_per_share_wei=500000000000000000,  # 0.5
                quantity_wei=100000000000000000000,  # 100 shares
            )
        )

        # BUY: makerAmount = price * qty / 1e18
        # 0.5 * 100 = 50 USDT
        assert amounts.maker_amount == 50000000000000000000
        assert amounts.taker_amount == 100000000000000000000

    def test_sell_order_basic(self, builder: OrderBuilder):
        """Test basic SELL order calculation."""
        amounts = builder.get_limit_order_amounts(
            LimitHelperInput(
                side=Side.SELL,
                price_per_share_wei=500000000000000000,  # 0.5
                quantity_wei=100000000000000000000,  # 100 shares
            )
        )

        # SELL: takerAmount = price * qty / 1e18
        # 0.5 * 100 = 50 USDT
        assert amounts.maker_amount == 100000000000000000000  # shares
        assert amounts.taker_amount == 50000000000000000000  # USDT

    def test_price_truncation(self, builder: OrderBuilder):
        """Test price truncation to 3 significant digits."""
        amounts = builder.get_limit_order_amounts(
            LimitHelperInput(
                side=Side.BUY,
                price_per_share_wei=123456789000000000,  # Should truncate to 123000000000000000
                quantity_wei=100000000000000000000,
            )
        )

        assert amounts.price_per_share == 123000000000000000

    def test_quantity_truncation(self, builder: OrderBuilder):
        """Test quantity truncation to 5 significant digits."""
        amounts = builder.get_limit_order_amounts(
            LimitHelperInput(
                side=Side.BUY,
                price_per_share_wei=500000000000000000,
                quantity_wei=123456789000000000000,  # Should truncate to 123450000000000000000
            )
        )

        assert amounts.taker_amount == 123450000000000000000

    def test_minimum_quantity(self, builder: OrderBuilder):
        """Test minimum quantity requirement (>= 1e16)."""
        # Should work with exactly 1e16
        amounts = builder.get_limit_order_amounts(
            LimitHelperInput(
                side=Side.BUY,
                price_per_share_wei=500000000000000000,
                quantity_wei=10000000000000000,  # 1e16
            )
        )
        assert amounts.taker_amount == 10000000000000000

    def test_below_minimum_quantity_raises(self, builder: OrderBuilder):
        """Test that below minimum quantity raises error."""
        with pytest.raises(InvalidQuantityError):
            builder.get_limit_order_amounts(
                LimitHelperInput(
                    side=Side.BUY,
                    price_per_share_wei=500000000000000000,
                    quantity_wei=9999999999999999,  # Just below 1e16
                )
            )


class TestMarketOrderAmounts:
    """Test market order amount calculations."""

    def test_market_buy_by_quantity(self, builder: OrderBuilder, fresh_orderbook: Book):
        """Test market BUY order by quantity."""
        amounts = builder.get_market_order_amounts(
            MarketHelperInput(
                side=Side.BUY,
                quantity_wei=50000000000000000000,  # 50 shares
            ),
            fresh_orderbook,
        )

        # Should consume from asks
        assert amounts.taker_amount > 0
        assert amounts.maker_amount > 0
        assert amounts.price_per_share > 0

    def test_market_sell_by_quantity(self, builder: OrderBuilder, fresh_orderbook: Book):
        """Test market SELL order by quantity."""
        amounts = builder.get_market_order_amounts(
            MarketHelperInput(
                side=Side.SELL,
                quantity_wei=50000000000000000000,  # 50 shares
            ),
            fresh_orderbook,
        )

        # Should consume from bids
        assert amounts.maker_amount > 0
        assert amounts.taker_amount > 0
        assert amounts.price_per_share > 0

    def test_market_buy_by_value(self, builder: OrderBuilder, fresh_orderbook: Book):
        """Test market BUY order by value."""
        amounts = builder.get_market_order_amounts(
            MarketHelperValueInput(
                side=Side.BUY,
                value_wei=10000000000000000000,  # 10 USDT
            ),
            fresh_orderbook,
        )

        # Should calculate shares based on value
        assert amounts.taker_amount > 0  # Number of shares
        assert amounts.maker_amount > 0  # Max spend

    def test_market_order_quantity_too_small(self, builder: OrderBuilder, fresh_orderbook: Book):
        """Test that too small quantity raises error."""
        with pytest.raises(InvalidQuantityError):
            builder.get_market_order_amounts(
                MarketHelperInput(
                    side=Side.BUY,
                    quantity_wei=1000,  # Too small
                ),
                fresh_orderbook,
            )

    def test_market_order_value_too_small(self, builder: OrderBuilder, fresh_orderbook: Book):
        """Test that too small value raises error."""
        with pytest.raises(InvalidQuantityError):
            builder.get_market_order_amounts(
                MarketHelperValueInput(
                    side=Side.BUY,
                    value_wei=100000000000000000,  # 0.1 USDT - too small (< 1e18)
                ),
                fresh_orderbook,
            )


class TestOrderAmountsConsistency:
    """Test consistency of order amount calculations."""

    def test_limit_buy_sell_symmetry(self, builder: OrderBuilder):
        """Test that BUY and SELL are symmetric for same price/quantity."""
        price = 500000000000000000  # 0.5
        qty = 100000000000000000000  # 100

        buy = builder.get_limit_order_amounts(
            LimitHelperInput(side=Side.BUY, price_per_share_wei=price, quantity_wei=qty)
        )

        sell = builder.get_limit_order_amounts(
            LimitHelperInput(side=Side.SELL, price_per_share_wei=price, quantity_wei=qty)
        )

        # For same price/qty, BUY's taker == SELL's maker (shares)
        # and BUY's maker == SELL's taker (USDT)
        assert buy.taker_amount == sell.maker_amount
        assert buy.maker_amount == sell.taker_amount

    def test_price_per_share_consistency(self, builder: OrderBuilder):
        """Test that price_per_share is consistent."""
        price = 333000000000000000  # 0.333 (after truncation: 0.333)
        qty = 100000000000000000000  # 100

        amounts = builder.get_limit_order_amounts(
            LimitHelperInput(side=Side.BUY, price_per_share_wei=price, quantity_wei=qty)
        )

        # Price per share should match input (after truncation)
        assert amounts.price_per_share == price
