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


class TestFloatingPointPrecision:
    """Test that floating-point precision errors are handled correctly."""

    def test_precision_bug_0_46(self, builder: OrderBuilder):
        """Test the specific case: 0.46 should convert exactly to wei."""
        book = Book(
            market_id=1,
            update_timestamp_ms=0,
            asks=[(0.46, 100.0)],
            bids=[(0.45, 100.0)],
        )

        amounts = builder.get_market_order_amounts(
            MarketHelperInput(side=Side.BUY, quantity_wei=int(10e18)),
            book,
        )

        # The last_price should be exactly 460000000000000000, not 460000000000000001
        assert amounts.last_price == 460000000000000000

    def test_precision_bug_0_421031(self, builder: OrderBuilder):
        """Test case from bug report: 0.421031 should convert exactly."""
        book = Book(
            market_id=1,
            update_timestamp_ms=0,
            asks=[(0.421031, 100.0)],
            bids=[(0.42, 100.0)],
        )

        amounts = builder.get_market_order_amounts(
            MarketHelperInput(side=Side.BUY, quantity_wei=int(10e18)),
            book,
        )

        # Should be exactly 421031000000000000, not 421030999999999936 or 421031000000000001
        assert amounts.last_price == 421031000000000000

    def test_precision_bug_0_07(self, builder: OrderBuilder):
        """Test case: 0.07 should convert exactly."""
        book = Book(
            market_id=1,
            update_timestamp_ms=0,
            asks=[(0.07, 100.0)],
            bids=[(0.06, 100.0)],
        )

        amounts = builder.get_market_order_amounts(
            MarketHelperInput(side=Side.BUY, quantity_wei=int(10e18)),
            book,
        )

        # Should be exactly 70000000000000000, not 70000000000000008
        assert amounts.last_price == 70000000000000000

    def test_precision_bug_0_009(self, builder: OrderBuilder):
        """Test case: 0.009 should convert exactly."""
        book = Book(
            market_id=1,
            update_timestamp_ms=0,
            asks=[(0.009, 500.0)],
            bids=[(0.008, 100.0)],
        )

        amounts = builder.get_market_order_amounts(
            MarketHelperInput(side=Side.BUY, quantity_wei=int(10e18)),
            book,
        )

        # Should be exactly 9000000000000000, not 8999999999999998
        assert amounts.last_price == 9000000000000000

    def test_problematic_decimals(self, builder: OrderBuilder):
        """Test various decimals known to cause floating-point issues."""
        problematic_prices = [
            (0.01, 10000000000000000),
            (0.03, 30000000000000000),
            (0.07, 70000000000000000),
            (0.11, 110000000000000000),
            (0.13, 130000000000000000),
            (0.17, 170000000000000000),
            (0.19, 190000000000000000),
            (0.23, 230000000000000000),
            (0.29, 290000000000000000),
            (0.31, 310000000000000000),
            (0.33, 330000000000000000),
            (0.37, 370000000000000000),
            (0.41, 410000000000000000),
            (0.43, 430000000000000000),
            (0.46, 460000000000000000),
            (0.47, 470000000000000000),
            (0.53, 530000000000000000),
            (0.59, 590000000000000000),
            (0.61, 610000000000000000),
            (0.67, 670000000000000000),
            (0.71, 710000000000000000),
            (0.73, 730000000000000000),
            (0.79, 790000000000000000),
            (0.83, 830000000000000000),
            (0.89, 890000000000000000),
            (0.97, 970000000000000000),
        ]

        for price, expected_wei in problematic_prices:
            book = Book(
                market_id=1,
                update_timestamp_ms=0,
                asks=[(price, 100.0)],
                bids=[(price - 0.01, 100.0)],
            )

            amounts = builder.get_market_order_amounts(
                MarketHelperInput(side=Side.BUY, quantity_wei=int(10e18)),
                book,
            )

            assert amounts.last_price == expected_wei, (
                f"Price {price} converted to {amounts.last_price}, expected {expected_wei}"
            )

    def test_original_bug_report_case(self, builder: OrderBuilder):
        """Test the exact case from the bug report."""
        book = Book(
            market_id=1,
            update_timestamp_ms=0,
            asks=[(0.46, 18.208), (0.48, 442.3), (0.48, 187.3)],
            bids=[(0.44, 36.77), (0.41, 474.1), (0.38, 328.03)],
        )

        amounts = builder.get_market_order_amounts(
            MarketHelperInput(side=Side.SELL, quantity_wei=100000000000000000000),
            book,
        )

        # price_per_share should be 421031000000000000, not 421031000000000001
        # This is the weighted average price calculation
        assert amounts.price_per_share == 421031000000000000
        assert amounts.last_price == 410000000000000000

    def test_precision_bug_0_777(self, builder: OrderBuilder):
        """Test case: 0.777 should convert exactly without intermediate precision loss."""
        book = Book(
            market_id=1,
            update_timestamp_ms=0,
            asks=[(0.777, 3.8769543979049894), (0.777, 411.8603781833764)],
            bids=[(0.69, 143.26520575527368), (0.51, 214.46972573717937)],
        )

        amounts = builder.get_market_order_amounts(
            MarketHelperInput(side=Side.BUY, quantity_wei=62430861279963832320),
            book,
        )

        # price_per_share should be 777000000000000000, not 776999999999999999
        # This tests the fix for intermediate division precision loss
        assert amounts.price_per_share == 777000000000000000
        assert amounts.last_price == 777000000000000000


class TestSlippage:
    """Test slippage application to market order amounts."""

    @pytest.fixture
    def slippage_book(self) -> Book:
        """Orderbook for slippage tests."""
        return Book(
            market_id=1,
            update_timestamp_ms=0,
            asks=[
                (0.27, 100.0),
                (0.30, 200.0),
            ],
            bids=[
                (0.27, 100.0),
                (0.25, 200.0),
            ],
        )

    def test_buy_by_quantity_inflates_maker_amount(
        self, builder: OrderBuilder, slippage_book: Book
    ):
        """BUY with slippage should inflate makerAmount (collateral offered)."""
        without = builder.get_market_order_amounts(
            MarketHelperInput(side=Side.BUY, quantity_wei=int(100e18)),
            slippage_book,
        )
        with_slippage = builder.get_market_order_amounts(
            MarketHelperInput(side=Side.BUY, quantity_wei=int(100e18), slippage_bps=500),
            slippage_book,
        )

        expected_maker = (without.maker_amount * 10_500) // 10_000
        assert with_slippage.maker_amount == expected_maker
        assert with_slippage.taker_amount == without.taker_amount
        assert with_slippage.price_per_share == without.price_per_share
        assert with_slippage.last_price == without.last_price
        assert with_slippage.slippage_bps == 500

    def test_sell_by_quantity_deflates_taker_amount(
        self, builder: OrderBuilder, slippage_book: Book
    ):
        """SELL with slippage should deflate takerAmount (collateral received)."""
        without = builder.get_market_order_amounts(
            MarketHelperInput(side=Side.SELL, quantity_wei=int(100e18)),
            slippage_book,
        )
        with_slippage = builder.get_market_order_amounts(
            MarketHelperInput(side=Side.SELL, quantity_wei=int(100e18), slippage_bps=500),
            slippage_book,
        )

        expected_taker = (without.taker_amount * 9_500) // 10_000
        assert with_slippage.taker_amount == expected_taker
        assert with_slippage.maker_amount == without.maker_amount
        assert with_slippage.price_per_share == without.price_per_share
        assert with_slippage.last_price == without.last_price
        assert with_slippage.slippage_bps == 500

    def test_buy_by_value_inflates_maker_amount(self, builder: OrderBuilder, slippage_book: Book):
        """BUY by value with slippage should inflate makerAmount."""
        without = builder.get_market_order_amounts(
            MarketHelperValueInput(side=Side.BUY, value_wei=int(10e18)),
            slippage_book,
        )
        with_slippage = builder.get_market_order_amounts(
            MarketHelperValueInput(side=Side.BUY, value_wei=int(10e18), slippage_bps=500),
            slippage_book,
        )

        expected_maker = (without.maker_amount * 10_500) // 10_000
        assert with_slippage.maker_amount == expected_maker
        assert with_slippage.taker_amount == without.taker_amount
        assert with_slippage.slippage_bps == 500

    def test_no_slippage_by_default(self, builder: OrderBuilder, slippage_book: Book):
        """Omitting slippage_bps should produce amounts identical to slippage_bps=0."""
        result = builder.get_market_order_amounts(
            MarketHelperInput(side=Side.BUY, quantity_wei=int(100e18)),
            slippage_book,
        )

        # makerAmount should equal lastPrice * qty / 1e18 (no slippage applied)
        expected_maker = (result.last_price * int(100e18)) // int(1e18)
        assert result.maker_amount == expected_maker
        assert result.slippage_bps == 0

    def test_explicit_zero_slippage_matches_default(
        self, builder: OrderBuilder, slippage_book: Book
    ):
        """slippage_bps=0 should produce identical results to omitting it."""
        default_result = builder.get_market_order_amounts(
            MarketHelperInput(side=Side.BUY, quantity_wei=int(100e18)),
            slippage_book,
        )
        zero_result = builder.get_market_order_amounts(
            MarketHelperInput(side=Side.BUY, quantity_wei=int(100e18), slippage_bps=0),
            slippage_book,
        )

        assert zero_result.maker_amount == default_result.maker_amount
        assert zero_result.taker_amount == default_result.taker_amount
        assert zero_result.slippage_bps == 0

    def test_buy_clamps_at_one_dollar_per_share(self, builder: OrderBuilder):
        """BUY makerAmount should be clamped at $1/share when slippage pushes price above $1."""
        high_price_book = Book(
            market_id=1,
            update_timestamp_ms=0,
            asks=[(0.97, 100.0)],
            bids=[(0.96, 100.0)],
        )

        result = builder.get_market_order_amounts(
            MarketHelperInput(
                side=Side.BUY, quantity_wei=int(100e18), slippage_bps=500
            ),  # 5% on 0.97 = 1.0185
            high_price_book,
        )

        # makerAmount should be clamped to takerAmount (shares), i.e. $1/share
        assert result.maker_amount == result.taker_amount

    def test_sell_floors_taker_amount_at_zero(self, builder: OrderBuilder):
        """SELL takerAmount should floor at 0 with extreme slippage."""
        book = Book(
            market_id=1,
            update_timestamp_ms=0,
            asks=[(0.50, 100.0)],
            bids=[(0.49, 100.0)],
        )

        result = builder.get_market_order_amounts(
            MarketHelperInput(side=Side.SELL, quantity_wei=int(100e18), slippage_bps=10_001),
            book,
        )

        assert result.taker_amount == 0


class TestSlippageDeepBook:
    """Test slippage with multi-tier books where avg != worst tier."""

    @pytest.fixture
    def deep_book(self) -> Book:
        """Book: avg BUY = 0.266, worst ask = 0.30; avg SELL ≈ 0.277, worst bid = 0.25."""
        return Book(
            market_id=1,
            update_timestamp_ms=0,
            asks=[
                (0.25, 50.0),
                (0.27, 30.0),
                (0.30, 20.0),
            ],
            bids=[
                (0.30, 50.0),
                (0.27, 30.0),
                (0.25, 20.0),
            ],
        )

    def test_buy_slippage_applies_to_worst_tier(self, builder: OrderBuilder, deep_book: Book):
        """Slippage buffer is applied to worst tier price (0.30)."""
        without = builder.get_market_order_amounts(
            MarketHelperInput(side=Side.BUY, quantity_wei=int(100e18)),
            deep_book,
        )
        with_slippage = builder.get_market_order_amounts(
            MarketHelperInput(side=Side.BUY, quantity_wei=int(100e18), slippage_bps=500),
            deep_book,
        )

        expected = (without.maker_amount * 10_500) // 10_000
        assert with_slippage.maker_amount == expected
        assert with_slippage.last_price == int(0.30e18)

    def test_sell_slippage_applies_to_worst_tier(self, builder: OrderBuilder, deep_book: Book):
        """Slippage buffer is applied to worst tier price (0.25)."""
        without = builder.get_market_order_amounts(
            MarketHelperInput(side=Side.SELL, quantity_wei=int(100e18)),
            deep_book,
        )
        with_slippage = builder.get_market_order_amounts(
            MarketHelperInput(side=Side.SELL, quantity_wei=int(100e18), slippage_bps=500),
            deep_book,
        )

        expected = (without.taker_amount * 9_500) // 10_000
        assert with_slippage.taker_amount == expected
        assert with_slippage.last_price == int(0.25e18)
