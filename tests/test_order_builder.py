"""Tests for the OrderBuilder class."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from predict_sdk import (
    BuildOrderInput,
    ChainId,
    InvalidExpirationError,
    InvalidQuantityError,
    LimitHelperInput,
    MakerSignerMismatchError,
    MissingSignerError,
    OrderBuilder,
    Side,
    SignatureType,
)


class TestOrderBuilderMake:
    """Test OrderBuilder factory method."""

    def test_make_without_signer(self):
        """Create OrderBuilder without signer for read-only operations."""
        builder = OrderBuilder.make(ChainId.BNB_MAINNET)
        assert builder.contracts is None

    def test_make_with_chain_id_mainnet(self):
        """Create OrderBuilder for mainnet."""
        builder = OrderBuilder.make(ChainId.BNB_MAINNET)
        assert builder is not None

    def test_make_with_chain_id_testnet(self):
        """Create OrderBuilder for testnet."""
        builder = OrderBuilder.make(ChainId.BNB_TESTNET)
        assert builder is not None


class TestBuildOrder:
    """Test order building functionality."""

    def test_build_limit_order(self, builder: OrderBuilder):
        """Build a limit order."""
        order = builder.build_order(
            "LIMIT",
            BuildOrderInput(
                side=Side.BUY,
                token_id="12345",
                maker_amount="1000000000000000000",
                taker_amount="2000000000000000000",
                fee_rate_bps=100,
            ),
        )

        assert order.side == Side.BUY
        assert order.token_id == "12345"
        assert order.maker_amount == "1000000000000000000"
        assert order.taker_amount == "2000000000000000000"
        assert order.fee_rate_bps == "100"
        assert order.signature_type == SignatureType.EOA

    def test_build_market_order(self, builder: OrderBuilder):
        """Build a market order."""
        order = builder.build_order(
            "MARKET",
            BuildOrderInput(
                side=Side.SELL,
                token_id="67890",
                maker_amount="500000000000000000",
                taker_amount="250000000000000000",
                fee_rate_bps=50,
            ),
        )

        assert order.side == Side.SELL
        assert order.token_id == "67890"

    def test_build_order_with_custom_salt(self, builder: OrderBuilder):
        """Build order with custom salt."""
        order = builder.build_order(
            "LIMIT",
            BuildOrderInput(
                side=Side.BUY,
                token_id="12345",
                maker_amount="1000000000000000000",
                taker_amount="2000000000000000000",
                fee_rate_bps=100,
                salt=123456789,
            ),
        )

        assert order.salt == "123456789"

    def test_build_order_with_expiration(self, builder: OrderBuilder):
        """Build order with custom expiration."""
        future_date = datetime(2100, 1, 1, tzinfo=timezone.utc)
        order = builder.build_order(
            "LIMIT",
            BuildOrderInput(
                side=Side.BUY,
                token_id="12345",
                maker_amount="1000000000000000000",
                taker_amount="2000000000000000000",
                fee_rate_bps=100,
                expires_at=future_date,
            ),
        )

        assert order.expiration == str(int(future_date.timestamp()))

    def test_build_order_past_expiration_raises(self, builder: OrderBuilder):
        """Building a LIMIT order with past expiration should raise."""
        past_date = datetime(2000, 1, 1, tzinfo=timezone.utc)
        with pytest.raises(InvalidExpirationError):
            builder.build_order(
                "LIMIT",
                BuildOrderInput(
                    side=Side.BUY,
                    token_id="12345",
                    maker_amount="1000000000000000000",
                    taker_amount="2000000000000000000",
                    fee_rate_bps=100,
                    expires_at=past_date,
                ),
            )


class TestLimitOrderAmounts:
    """Test limit order amount calculations."""

    def test_buy_order_amounts(self, builder: OrderBuilder):
        """Calculate amounts for a buy order."""
        amounts = builder.get_limit_order_amounts(
            LimitHelperInput(
                side=Side.BUY,
                price_per_share_wei=400000000000000000,  # 0.4
                quantity_wei=10000000000000000000,  # 10 shares
            )
        )

        # BUY: makerAmount = price * qty / precision
        # 0.4 * 10 = 4 USDT
        assert amounts.maker_amount == 4000000000000000000
        assert amounts.taker_amount == 10000000000000000000
        assert amounts.price_per_share == 400000000000000000

    def test_sell_order_amounts(self, builder: OrderBuilder):
        """Calculate amounts for a sell order."""
        amounts = builder.get_limit_order_amounts(
            LimitHelperInput(
                side=Side.SELL,
                price_per_share_wei=600000000000000000,  # 0.6
                quantity_wei=5000000000000000000,  # 5 shares
            )
        )

        # SELL: takerAmount = price * qty / precision
        # 0.6 * 5 = 3 USDT
        assert amounts.maker_amount == 5000000000000000000  # shares offered
        assert amounts.taker_amount == 3000000000000000000  # USDT to receive
        assert amounts.price_per_share == 600000000000000000

    def test_invalid_quantity_raises(self, builder: OrderBuilder):
        """Raise error for invalid quantity."""
        with pytest.raises(InvalidQuantityError):
            builder.get_limit_order_amounts(
                LimitHelperInput(
                    side=Side.BUY,
                    price_per_share_wei=400000000000000000,
                    quantity_wei=1000,  # Too small (< 1e16)
                )
            )

    def test_significant_digit_truncation(self, builder: OrderBuilder):
        """Test that values are truncated to significant digits."""
        # Price should be truncated to 3 significant digits
        # Quantity should be truncated to 5 significant digits
        amounts = builder.get_limit_order_amounts(
            LimitHelperInput(
                side=Side.BUY,
                price_per_share_wei=123456789000000000,  # Should truncate to 123000000000000000
                quantity_wei=12345678900000000000,  # Should truncate to 12345000000000000000
            )
        )

        # Verify truncation happened
        # The exact values depend on retainSignificantDigits implementation
        assert amounts.price_per_share == 123000000000000000
        assert amounts.taker_amount == 12345000000000000000


class TestTypedData:
    """Test EIP-712 typed data generation."""

    def test_build_typed_data(self, builder: OrderBuilder):
        """Build typed data for an order."""
        order = builder.build_order(
            "LIMIT",
            BuildOrderInput(
                side=Side.BUY,
                token_id="12345",
                maker_amount="1000000000000000000",
                taker_amount="2000000000000000000",
                fee_rate_bps=100,
            ),
        )

        typed_data = builder.build_typed_data(
            order,
            is_neg_risk=False,
            is_yield_bearing=False,
        )

        assert typed_data.primary_type == "Order"
        assert typed_data.domain["name"] == "predict.fun CTF Exchange"
        assert typed_data.domain["version"] == "1"
        assert typed_data.domain["chainId"] == ChainId.BNB_MAINNET
        assert "Order" in typed_data.types
        assert "EIP712Domain" in typed_data.types

    def test_build_typed_data_neg_risk(self, builder: OrderBuilder):
        """Build typed data for a NegRisk order."""
        order = builder.build_order(
            "LIMIT",
            BuildOrderInput(
                side=Side.BUY,
                token_id="12345",
                maker_amount="1000000000000000000",
                taker_amount="2000000000000000000",
                fee_rate_bps=100,
            ),
        )

        typed_data = builder.build_typed_data(
            order,
            is_neg_risk=True,
            is_yield_bearing=False,
        )

        # NegRisk should use a different verifying contract
        assert typed_data.domain["verifyingContract"] is not None


class TestSignature:
    """Test order signing functionality."""

    def test_sign_without_signer_raises(self, builder: OrderBuilder):
        """Signing without a signer should raise MissingSignerError."""
        order = builder.build_order(
            "LIMIT",
            BuildOrderInput(
                side=Side.BUY,
                token_id="12345",
                maker_amount="1000000000000000000",
                taker_amount="2000000000000000000",
                fee_rate_bps=100,
            ),
        )

        typed_data = builder.build_typed_data(
            order,
            is_neg_risk=False,
            is_yield_bearing=False,
        )

        with pytest.raises(MissingSignerError):
            builder.sign_typed_data_order(typed_data)


class TestContractInteractions:
    """Test contract interaction methods."""

    def test_balance_of_without_signer_raises(self, builder: OrderBuilder):
        """balance_of without signer should raise MissingSignerError."""
        with pytest.raises(MissingSignerError):
            builder.balance_of()

    def test_set_approvals_without_signer_raises(self, builder: OrderBuilder):
        """set_approvals without signer should raise MissingSignerError."""
        with pytest.raises(MissingSignerError):
            builder.set_approvals()
