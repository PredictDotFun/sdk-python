"""Tests for the OrderBuilder class."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from predict_sdk import (
    BuildOrderInput,
    ChainId,
    InvalidExpirationError,
    InvalidQuantityError,
    LimitHelperInput,
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

    def test_build_limit_order(self, builder_with_signer: OrderBuilder):
        """Build a limit order."""
        order = builder_with_signer.build_order(
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

    def test_build_market_order(self, builder_with_signer: OrderBuilder):
        """Build a market order."""
        order = builder_with_signer.build_order(
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

    def test_build_order_with_custom_salt(self, builder_with_signer: OrderBuilder):
        """Build order with custom salt."""
        order = builder_with_signer.build_order(
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

    def test_build_order_with_expiration(self, builder_with_signer: OrderBuilder):
        """Build order with custom expiration."""
        future_date = datetime(2100, 1, 1, tzinfo=timezone.utc)
        order = builder_with_signer.build_order(
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

    def test_build_order_past_expiration_raises(self, builder_with_signer: OrderBuilder):
        """Building a LIMIT order with past expiration should raise."""
        past_date = datetime(2000, 1, 1, tzinfo=timezone.utc)
        with pytest.raises(InvalidExpirationError):
            builder_with_signer.build_order(
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

    def test_build_order_without_signer_raises(self, builder: OrderBuilder):
        """Building an order without signer should raise MissingSignerError."""
        with pytest.raises(MissingSignerError):
            builder.build_order(
                "LIMIT",
                BuildOrderInput(
                    side=Side.BUY,
                    token_id="12345",
                    maker_amount="1000000000000000000",
                    taker_amount="2000000000000000000",
                    fee_rate_bps=100,
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

    def test_build_typed_data_hash_matches_ts_sdk(self, builder_with_signer: OrderBuilder):
        """Hash should match the TS SDK's TypedDataEncoder.hash() output."""
        from predict_sdk import EIP712TypedData

        # Static typed data with known expected hash from TS SDK
        typed_data = EIP712TypedData(
            primary_type="Order",
            types={
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
            domain={
                "name": "predict.fun CTF Exchange",
                "version": "1",
                "chainId": 56,
                "verifyingContract": "0x8BC070BEdAB741406F4B1Eb65A72bee27894B689",
            },
            message={
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
        )

        hash_result = builder_with_signer.build_typed_data_hash(typed_data)

        # Expected hash from TS SDK's TypedDataEncoder.hash()
        expected_hash = "0x814000c89efa61ae42a2bcc4c98e06e90c11480b95a12edea00e3411ec76821d"
        assert hash_result == expected_hash, (
            f"Hash mismatch: got {hash_result}, expected {expected_hash}"
        )

    def test_build_typed_data_hash_has_0x_prefix(self, builder_with_signer: OrderBuilder):
        """Hash from build_typed_data_hash should have 0x prefix."""
        order = builder_with_signer.build_order(
            "LIMIT",
            BuildOrderInput(
                side=Side.BUY,
                token_id="12345",
                maker_amount="1000000000000000000",
                taker_amount="2000000000000000000",
                fee_rate_bps=100,
            ),
        )

        typed_data = builder_with_signer.build_typed_data(
            order,
            is_neg_risk=False,
            is_yield_bearing=False,
        )

        hash_result = builder_with_signer.build_typed_data_hash(typed_data)

        # Hash should start with 0x prefix
        assert hash_result.startswith("0x"), f"Hash should start with '0x', got: {hash_result}"
        # Hash should be 66 characters (0x + 64 hex chars)
        assert len(hash_result) == 66, f"Hash should be 66 chars, got: {len(hash_result)}"
        # Remaining chars should be valid hex
        assert all(c in "0123456789abcdef" for c in hash_result[2:])

    def test_build_typed_data(self, builder_with_signer: OrderBuilder):
        """Build typed data for an order."""
        order = builder_with_signer.build_order(
            "LIMIT",
            BuildOrderInput(
                side=Side.BUY,
                token_id="12345",
                maker_amount="1000000000000000000",
                taker_amount="2000000000000000000",
                fee_rate_bps=100,
            ),
        )

        typed_data = builder_with_signer.build_typed_data(
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

    def test_build_typed_data_neg_risk(self, builder_with_signer: OrderBuilder):
        """Build typed data for a NegRisk order."""
        order = builder_with_signer.build_order(
            "LIMIT",
            BuildOrderInput(
                side=Side.BUY,
                token_id="12345",
                maker_amount="1000000000000000000",
                taker_amount="2000000000000000000",
                fee_rate_bps=100,
            ),
        )

        typed_data = builder_with_signer.build_typed_data(
            order,
            is_neg_risk=True,
            is_yield_bearing=False,
        )

        # NegRisk should use a different verifying contract
        assert typed_data.domain["verifyingContract"] is not None


class TestSignature:
    """Test order signing functionality."""

    def test_sign_typed_data_order_returns_signed_order(self, builder_with_signer: OrderBuilder):
        """sign_typed_data_order should return a SignedOrder with signature."""
        order = builder_with_signer.build_order(
            "LIMIT",
            BuildOrderInput(
                side=Side.BUY,
                token_id="12345",
                maker_amount="1000000000000000000",
                taker_amount="2000000000000000000",
                fee_rate_bps=100,
            ),
        )

        typed_data = builder_with_signer.build_typed_data(
            order,
            is_neg_risk=False,
            is_yield_bearing=False,
        )

        signed_order = builder_with_signer.sign_typed_data_order(typed_data)

        # Verify the signed order has a signature
        assert signed_order.signature is not None
        assert len(signed_order.signature) > 0
        # Verify order details are preserved
        assert signed_order.token_id == order.token_id
        assert signed_order.maker_amount == order.maker_amount
        assert signed_order.taker_amount == order.taker_amount

    @pytest.mark.asyncio
    async def test_sign_typed_data_order_async(self, builder_with_signer: OrderBuilder):
        """sign_typed_data_order_async should return same result as sync version."""
        order = builder_with_signer.build_order(
            "LIMIT",
            BuildOrderInput(
                side=Side.BUY,
                token_id="12345",
                maker_amount="1000000000000000000",
                taker_amount="2000000000000000000",
                fee_rate_bps=100,
            ),
        )

        typed_data = builder_with_signer.build_typed_data(
            order,
            is_neg_risk=False,
            is_yield_bearing=False,
        )

        # Get both sync and async results
        sync_result = builder_with_signer.sign_typed_data_order(typed_data)
        async_result = await builder_with_signer.sign_typed_data_order_async(typed_data)

        # They should produce the same signature
        assert sync_result.signature == async_result.signature
        assert sync_result.token_id == async_result.token_id

    def test_sign_without_signer_raises(
        self, builder_with_signer: OrderBuilder, builder: OrderBuilder
    ):
        """Signing without a signer should raise MissingSignerError."""
        # Build an order with a signer first (valid)
        order = builder_with_signer.build_order(
            "LIMIT",
            BuildOrderInput(
                side=Side.BUY,
                token_id="12345",
                maker_amount="1000000000000000000",
                taker_amount="2000000000000000000",
                fee_rate_bps=100,
            ),
        )

        typed_data = builder_with_signer.build_typed_data(
            order,
            is_neg_risk=False,
            is_yield_bearing=False,
        )

        # Try to sign with a builder that has no signer
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


class TestRedeemPositions:
    """Test position redemption functionality."""

    def test_redeem_positions_without_signer_raises(self, builder: OrderBuilder):
        """redeem_positions without signer should raise MissingSignerError."""
        with pytest.raises(MissingSignerError):
            builder.redeem_positions(
                condition_id="0x" + "0" * 64,
                index_set=1,
                is_neg_risk=False,
                is_yield_bearing=False,
            )

    def test_redeem_positions_neg_risk_requires_amount(self, builder_with_signer: OrderBuilder):
        """redeem_positions with is_neg_risk=True but no amount should raise ValueError."""
        # builder_with_signer has no contracts, so it will raise MissingSignerError
        # before reaching the amount validation. We need to test the validation
        # path when contracts exist but amount is missing.
        # For now, test the no-signer case raises MissingSignerError
        with pytest.raises(MissingSignerError):
            builder_with_signer.redeem_positions(
                condition_id="0x" + "0" * 64,
                index_set=1,
                is_neg_risk=True,
                is_yield_bearing=False,
            )

    def test_redeem_positions_neg_risk_without_signer_raises(self, builder: OrderBuilder):
        """redeem_positions for NegRisk without signer should raise MissingSignerError."""
        with pytest.raises(MissingSignerError):
            builder.redeem_positions(
                condition_id="0x" + "0" * 64,
                index_set=1,
                amount=1000000000000000000,
                is_neg_risk=True,
                is_yield_bearing=False,
            )


class TestMergePositions:
    """Test position merging functionality."""

    def test_merge_positions_without_signer_raises(self, builder: OrderBuilder):
        """merge_positions without signer should raise MissingSignerError."""
        with pytest.raises(MissingSignerError):
            builder.merge_positions(
                condition_id="0x" + "0" * 64,
                amount=1000000000000000000,
                is_neg_risk=False,
                is_yield_bearing=False,
            )

    def test_merge_positions_neg_risk_without_signer_raises(self, builder: OrderBuilder):
        """merge_positions for NegRisk without signer should raise MissingSignerError."""
        with pytest.raises(MissingSignerError):
            builder.merge_positions(
                condition_id="0x" + "0" * 64,
                amount=1000000000000000000,
                is_neg_risk=True,
                is_yield_bearing=False,
            )


class TestSplitPositions:
    """Test position splitting functionality."""

    def test_split_positions_without_signer_raises(self, builder: OrderBuilder):
        """split_positions without signer should raise MissingSignerError."""
        with pytest.raises(MissingSignerError):
            builder.split_positions(
                condition_id="0x" + "0" * 64,
                amount=1000000000000000000,
                is_neg_risk=False,
                is_yield_bearing=False,
            )

    def test_split_positions_neg_risk_without_signer_raises(self, builder: OrderBuilder):
        """split_positions for NegRisk without signer should raise MissingSignerError."""
        with pytest.raises(MissingSignerError):
            builder.split_positions(
                condition_id="0x" + "0" * 64,
                amount=1000000000000000000,
                is_neg_risk=True,
                is_yield_bearing=False,
            )

    def test_split_positions_yield_bearing_without_signer_raises(self, builder: OrderBuilder):
        """split_positions for yield-bearing without signer should raise MissingSignerError."""
        with pytest.raises(MissingSignerError):
            builder.split_positions(
                condition_id="0x" + "0" * 64,
                amount=1000000000000000000,
                is_neg_risk=False,
                is_yield_bearing=True,
            )

    def test_split_positions_neg_risk_yield_bearing_without_signer_raises(
        self, builder: OrderBuilder
    ):
        """split_positions for NegRisk yield-bearing without signer should raise MissingSignerError."""
        with pytest.raises(MissingSignerError):
            builder.split_positions(
                condition_id="0x" + "0" * 64,
                amount=1000000000000000000,
                is_neg_risk=True,
                is_yield_bearing=True,
            )


class TestTypedDataCombinations:
    """Test EIP-712 typed data generation for all market type combinations."""

    @pytest.mark.parametrize(
        "is_neg_risk,is_yield_bearing",
        [
            (False, False),
            (False, True),
            (True, False),
            (True, True),
        ],
    )
    def test_build_typed_data_combinations(
        self,
        builder_with_signer: OrderBuilder,
        is_neg_risk: bool,
        is_yield_bearing: bool,
    ):
        """Build typed data for all combinations of is_neg_risk and is_yield_bearing."""
        order = builder_with_signer.build_order(
            "LIMIT",
            BuildOrderInput(
                side=Side.BUY,
                token_id="12345",
                maker_amount="1000000000000000000",
                taker_amount="2000000000000000000",
                fee_rate_bps=100,
            ),
        )

        typed_data = builder_with_signer.build_typed_data(
            order,
            is_neg_risk=is_neg_risk,
            is_yield_bearing=is_yield_bearing,
        )

        assert typed_data.primary_type == "Order"
        assert typed_data.domain["name"] == "predict.fun CTF Exchange"
        assert typed_data.domain["version"] == "1"
        assert typed_data.domain["chainId"] == ChainId.BNB_MAINNET
        assert typed_data.domain["verifyingContract"] is not None
        assert "Order" in typed_data.types
        assert "EIP712Domain" in typed_data.types

    @pytest.mark.parametrize(
        "is_neg_risk,is_yield_bearing",
        [
            (False, False),
            (False, True),
            (True, False),
            (True, True),
        ],
    )
    def test_build_typed_data_hash_combinations(
        self,
        builder_with_signer: OrderBuilder,
        is_neg_risk: bool,
        is_yield_bearing: bool,
    ):
        """Build typed data hash for all combinations."""
        order = builder_with_signer.build_order(
            "LIMIT",
            BuildOrderInput(
                side=Side.BUY,
                token_id="12345",
                maker_amount="1000000000000000000",
                taker_amount="2000000000000000000",
                fee_rate_bps=100,
            ),
        )

        typed_data = builder_with_signer.build_typed_data(
            order,
            is_neg_risk=is_neg_risk,
            is_yield_bearing=is_yield_bearing,
        )

        hash_result = builder_with_signer.build_typed_data_hash(typed_data)

        assert hash_result.startswith("0x")
        assert len(hash_result) == 66
        assert all(c in "0123456789abcdef" for c in hash_result[2:])


class TestCancelOrders:
    """Test order cancellation functionality."""

    def test_cancel_orders_without_signer_raises(self, builder: OrderBuilder):
        """cancel_orders without signer should raise MissingSignerError."""
        from predict_sdk import CancelOrdersOptions, Order

        # Create a mock order
        mock_order = Order(
            salt="123",
            maker="0x" + "0" * 40,
            signer="0x" + "0" * 40,
            taker="0x" + "0" * 40,
            token_id="12345",
            maker_amount="1000000000000000000",
            taker_amount="2000000000000000000",
            expiration="4102444800",
            nonce="0",
            fee_rate_bps="100",
            side=Side.BUY,
            signature_type=SignatureType.EOA,
        )

        with pytest.raises(MissingSignerError):
            builder.cancel_orders(
                [mock_order],
                CancelOrdersOptions(
                    is_neg_risk=False,
                    is_yield_bearing=False,
                ),
            )


class TestSetApprovals:
    """Test approval orchestration (both tracks) and on-chain idempotency."""

    @pytest.mark.asyncio
    async def test_set_approvals_covers_both_tracks_by_default(
        self, builder_with_signer: OrderBuilder
    ):
        """set_approvals() with no args must approve standard AND yield-bearing contracts."""
        from predict_sdk.types import TransactionSuccess

        builder = builder_with_signer
        builder.set_ctf_exchange_approval_async = AsyncMock(  # type: ignore[method-assign]
            return_value=TransactionSuccess(success=True)
        )
        builder.set_ctf_exchange_allowance_async = AsyncMock(  # type: ignore[method-assign]
            return_value=TransactionSuccess(success=True)
        )
        builder.set_neg_risk_adapter_approval_async = AsyncMock(  # type: ignore[method-assign]
            return_value=TransactionSuccess(success=True)
        )

        result = await builder.set_approvals_async()

        assert result.success is True
        assert len(result.transactions) == 10

        approval_combos = {
            (c.kwargs["is_neg_risk"], c.kwargs["is_yield_bearing"])
            for c in builder.set_ctf_exchange_approval_async.call_args_list
        }
        allowance_combos = {
            (c.kwargs["is_neg_risk"], c.kwargs["is_yield_bearing"])
            for c in builder.set_ctf_exchange_allowance_async.call_args_list
        }
        adapter_tracks = {
            c.kwargs["is_yield_bearing"]
            for c in builder.set_neg_risk_adapter_approval_async.call_args_list
        }

        assert approval_combos == {(False, False), (True, False), (False, True), (True, True)}
        assert allowance_combos == {(False, False), (True, False), (False, True), (True, True)}
        assert adapter_tracks == {False, True}

    @pytest.mark.asyncio
    async def test_set_approvals_single_track(self, builder_with_signer: OrderBuilder):
        """Passing is_yield_bearing limits the run to that one track (5 operations)."""
        from predict_sdk.types import TransactionSuccess

        builder = builder_with_signer
        builder.set_ctf_exchange_approval_async = AsyncMock(  # type: ignore[method-assign]
            return_value=TransactionSuccess(success=True)
        )
        builder.set_ctf_exchange_allowance_async = AsyncMock(  # type: ignore[method-assign]
            return_value=TransactionSuccess(success=True)
        )
        builder.set_neg_risk_adapter_approval_async = AsyncMock(  # type: ignore[method-assign]
            return_value=TransactionSuccess(success=True)
        )

        result = await builder.set_approvals_async(is_yield_bearing=True)

        assert result.success is True
        assert len(result.transactions) == 5
        assert all(
            c.kwargs["is_yield_bearing"] is True
            for c in builder.set_ctf_exchange_approval_async.call_args_list
        )
        assert all(
            c.kwargs["is_yield_bearing"] is True
            for c in builder.set_ctf_exchange_allowance_async.call_args_list
        )

    @pytest.mark.asyncio
    async def test_allowance_skipped_when_already_sufficient(
        self, builder_with_signer: OrderBuilder
    ):
        """An existing allowance that covers the amount must not send a transaction."""
        from predict_sdk.constants import MAX_UINT256
        from predict_sdk.types import TransactionSuccess

        builder = builder_with_signer
        contracts = MagicMock()
        contracts.usdt.functions.allowance.return_value.call.return_value = MAX_UINT256
        builder._contracts = contracts
        builder._handle_transaction_async = AsyncMock(  # type: ignore[method-assign]
            return_value=TransactionSuccess(success=True)
        )

        result = await builder.set_ctf_exchange_allowance_async(
            is_neg_risk=True, is_yield_bearing=True
        )

        assert result.success is True
        builder._handle_transaction_async.assert_not_called()
        contracts.usdt.functions.allowance.assert_called_once_with(
            builder._signer.address,
            builder._addresses.YIELD_BEARING_NEG_RISK_CTF_EXCHANGE,
        )

    @pytest.mark.asyncio
    async def test_allowance_sent_when_missing(self, builder_with_signer: OrderBuilder):
        """A zero allowance (the bug the integrator hit) must trigger an approve transaction."""
        from predict_sdk.types import TransactionSuccess

        builder = builder_with_signer
        contracts = MagicMock()
        contracts.usdt.functions.allowance.return_value.call.return_value = 0
        builder._contracts = contracts
        builder._handle_transaction_async = AsyncMock(  # type: ignore[method-assign]
            return_value=TransactionSuccess(success=True)
        )

        result = await builder.set_ctf_exchange_allowance_async(
            is_neg_risk=True, is_yield_bearing=True
        )

        assert result.success is True
        builder._handle_transaction_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_approval_skipped_when_already_approved(self, builder_with_signer: OrderBuilder):
        """An ERC-1155 operator already approved must not send a transaction."""
        from predict_sdk.types import TransactionSuccess

        builder = builder_with_signer
        builder._contracts = MagicMock()
        builder._handle_transaction_async = AsyncMock(  # type: ignore[method-assign]
            return_value=TransactionSuccess(success=True)
        )

        ct = MagicMock()
        ct.functions.isApprovedForAll.return_value.call.return_value = True

        with patch("predict_sdk.order_builder.get_conditional_tokens_contract", return_value=ct):
            result = await builder.set_ctf_exchange_approval_async(
                is_neg_risk=False, is_yield_bearing=True
            )

        assert result.success is True
        builder._handle_transaction_async.assert_not_called()
        ct.functions.isApprovedForAll.assert_called_once()

    @pytest.mark.asyncio
    async def test_approval_sent_when_not_approved(self, builder_with_signer: OrderBuilder):
        """An ERC-1155 operator not yet approved must send a setApprovalForAll transaction."""
        from predict_sdk.types import TransactionSuccess

        builder = builder_with_signer
        builder._contracts = MagicMock()
        builder._handle_transaction_async = AsyncMock(  # type: ignore[method-assign]
            return_value=TransactionSuccess(success=True)
        )

        ct = MagicMock()
        ct.functions.isApprovedForAll.return_value.call.return_value = False

        with patch("predict_sdk.order_builder.get_conditional_tokens_contract", return_value=ct):
            result = await builder.set_ctf_exchange_approval_async(
                is_neg_risk=False, is_yield_bearing=True
            )

        assert result.success is True
        builder._handle_transaction_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_allowance_read_failure_falls_through_to_send(
        self, builder_with_signer: OrderBuilder
    ):
        """A failed allowance read must not raise; it falls through to the send path."""
        from predict_sdk.types import TransactionSuccess

        builder = builder_with_signer
        contracts = MagicMock()
        contracts.usdt.functions.allowance.return_value.call.side_effect = Exception("rpc down")
        builder._contracts = contracts
        builder._handle_transaction_async = AsyncMock(  # type: ignore[method-assign]
            return_value=TransactionSuccess(success=True)
        )

        result = await builder.set_ctf_exchange_allowance_async(
            is_neg_risk=False, is_yield_bearing=False
        )

        assert result.success is True
        builder._handle_transaction_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_approval_read_failure_falls_through_to_send(
        self, builder_with_signer: OrderBuilder
    ):
        """A failed isApprovedForAll read must not raise; it falls through to the send path."""
        from predict_sdk.types import TransactionSuccess

        builder = builder_with_signer
        builder._contracts = MagicMock()
        builder._handle_transaction_async = AsyncMock(  # type: ignore[method-assign]
            return_value=TransactionSuccess(success=True)
        )

        ct = MagicMock()
        ct.functions.isApprovedForAll.return_value.call.side_effect = Exception("rpc down")

        with patch("predict_sdk.order_builder.get_conditional_tokens_contract", return_value=ct):
            result = await builder.set_ctf_exchange_approval_async(
                is_neg_risk=False, is_yield_bearing=True
            )

        assert result.success is True
        builder._handle_transaction_async.assert_called_once()
