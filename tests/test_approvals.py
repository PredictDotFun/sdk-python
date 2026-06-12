"""Tests for the scoped-approvals API."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from predict_sdk import (
    ApprovalProgress,
    ApprovalScope,
    ChainId,
    OrderBuilder,
    Side,
)
from predict_sdk.constants import ADDRESSES_BY_CHAIN_ID, MAX_INT256, MAX_UINT256
from predict_sdk.errors import InvalidApprovalOperationError, MissingSignerError
from predict_sdk.types import ApprovalStep, TransactionFail, TransactionSuccess

ADDRESSES = ADDRESSES_BY_CHAIN_ID[ChainId.BNB_MAINNET]


def ids(steps: list[ApprovalStep]) -> list[str]:
    return [s.id for s in steps]


class TestGetApprovalStepsTrade:
    def test_standard_both_sides(self, builder: OrderBuilder):
        steps = builder.get_approval_steps(
            ApprovalScope(operation="TRADE", is_neg_risk=False, is_yield_bearing=False)
        )
        assert ids(steps) == ["ERC1155_APPROVAL:CTF_EXCHANGE", "ERC20_ALLOWANCE:CTF_EXCHANGE"]

    def test_standard_buy_only_allowance(self, builder: OrderBuilder):
        steps = builder.get_approval_steps(
            ApprovalScope(
                operation="TRADE", is_neg_risk=False, is_yield_bearing=False, side=Side.BUY
            )
        )
        assert ids(steps) == ["ERC20_ALLOWANCE:CTF_EXCHANGE"]

    def test_standard_sell_only_approval(self, builder: OrderBuilder):
        steps = builder.get_approval_steps(
            ApprovalScope(
                operation="TRADE", is_neg_risk=False, is_yield_bearing=False, side=Side.SELL
            )
        )
        assert ids(steps) == ["ERC1155_APPROVAL:CTF_EXCHANGE"]

    def test_populates_spender_token_and_copy(self, builder: OrderBuilder):
        approval, allowance = builder.get_approval_steps(
            ApprovalScope(operation="TRADE", is_neg_risk=False, is_yield_bearing=False)
        )
        assert approval.type == "ERC1155_APPROVAL"
        assert approval.spender == ADDRESSES.CTF_EXCHANGE
        assert approval.token == ADDRESSES.CONDITIONAL_TOKENS
        assert approval.label == "Approve Exchange"
        assert approval.description == "Allows you to interact with the exchange."

        assert allowance.type == "ERC20_ALLOWANCE"
        assert allowance.spender == ADDRESSES.CTF_EXCHANGE
        assert allowance.token == ADDRESSES.USDT
        assert allowance.label == "Exchange Allowance"

    def test_neg_risk_includes_adapter_regardless_of_side(self, builder: OrderBuilder):
        both = builder.get_approval_steps(
            ApprovalScope(operation="TRADE", is_neg_risk=True, is_yield_bearing=False)
        )
        assert ids(both) == [
            "ERC1155_APPROVAL:NEG_RISK_CTF_EXCHANGE",
            "ERC1155_APPROVAL:NEG_RISK_ADAPTER",
            "ERC20_ALLOWANCE:NEG_RISK_CTF_EXCHANGE",
        ]

        buy = builder.get_approval_steps(
            ApprovalScope(
                operation="TRADE", is_neg_risk=True, is_yield_bearing=False, side=Side.BUY
            )
        )
        assert ids(buy) == [
            "ERC1155_APPROVAL:NEG_RISK_ADAPTER",
            "ERC20_ALLOWANCE:NEG_RISK_CTF_EXCHANGE",
        ]

        sell = builder.get_approval_steps(
            ApprovalScope(
                operation="TRADE", is_neg_risk=True, is_yield_bearing=False, side=Side.SELL
            )
        )
        assert ids(sell) == [
            "ERC1155_APPROVAL:NEG_RISK_CTF_EXCHANGE",
            "ERC1155_APPROVAL:NEG_RISK_ADAPTER",
        ]


class TestGetApprovalStepsOther:
    def test_split_standard(self, builder: OrderBuilder):
        steps = builder.get_approval_steps(
            ApprovalScope(operation="SPLIT", is_neg_risk=False, is_yield_bearing=False)
        )
        assert ids(steps) == ["ERC20_ALLOWANCE:CONDITIONAL_TOKENS"]
        assert steps[0].spender == ADDRESSES.CONDITIONAL_TOKENS
        assert steps[0].token == ADDRESSES.USDT
        assert steps[0].label == "Split Allowance"

    def test_split_neg_risk(self, builder: OrderBuilder):
        steps = builder.get_approval_steps(
            ApprovalScope(operation="SPLIT", is_neg_risk=True, is_yield_bearing=False)
        )
        assert ids(steps) == ["ERC20_ALLOWANCE:NEG_RISK_ADAPTER"]
        assert steps[0].label == "Multi-Outcome Split Allowance"

    def test_standard_merge_and_redeem_need_nothing(self, builder: OrderBuilder):
        assert (
            builder.get_approval_steps(
                ApprovalScope(operation="MERGE", is_neg_risk=False, is_yield_bearing=False)
            )
            == []
        )
        assert (
            builder.get_approval_steps(
                ApprovalScope(operation="REDEEM", is_neg_risk=False, is_yield_bearing=False)
            )
            == []
        )

    def test_neg_risk_merge_redeem_convert_need_adapter(self, builder: OrderBuilder):
        for op in ("MERGE", "REDEEM", "CONVERT"):
            steps = builder.get_approval_steps(
                ApprovalScope(operation=op, is_neg_risk=True, is_yield_bearing=False)  # type: ignore[arg-type]
            )
            assert ids(steps) == ["ERC1155_APPROVAL:NEG_RISK_ADAPTER"]

    def test_convert_on_standard_raises(self, builder: OrderBuilder):
        with pytest.raises(InvalidApprovalOperationError):
            builder.get_approval_steps(
                ApprovalScope(operation="CONVERT", is_neg_risk=False, is_yield_bearing=False)
            )

    def test_yield_bearing_keys_share_copy(self, builder: OrderBuilder):
        standard = builder.get_approval_steps(
            ApprovalScope(operation="TRADE", is_neg_risk=False, is_yield_bearing=False)
        )
        yield_bearing = builder.get_approval_steps(
            ApprovalScope(operation="TRADE", is_neg_risk=False, is_yield_bearing=True)
        )
        assert ids(yield_bearing) == [
            "ERC1155_APPROVAL:YIELD_BEARING_CTF_EXCHANGE",
            "ERC20_ALLOWANCE:YIELD_BEARING_CTF_EXCHANGE",
        ]
        # Copy is role-based, so it matches the standard track.
        assert [s.label for s in yield_bearing] == [s.label for s in standard]
        assert yield_bearing[0].spender == ADDRESSES.YIELD_BEARING_CTF_EXCHANGE


class TestCheckApproval:
    @pytest.mark.asyncio
    async def test_erc20_satisfied_by_allowance(self, builder_with_signer: OrderBuilder):
        builder = builder_with_signer
        contracts = MagicMock()
        contracts.usdt.functions.allowance.return_value.call.return_value = MAX_INT256
        builder._contracts = contracts

        step = builder.get_approval_steps(
            ApprovalScope(operation="SPLIT", is_neg_risk=False, is_yield_bearing=False)
        )[0]
        assert await builder.check_approval_async(step) is True

    @pytest.mark.asyncio
    async def test_erc20_below_threshold_unsatisfied(self, builder_with_signer: OrderBuilder):
        builder = builder_with_signer
        contracts = MagicMock()
        contracts.usdt.functions.allowance.return_value.call.return_value = MAX_INT256 - 1
        builder._contracts = contracts

        step = builder.get_approval_steps(
            ApprovalScope(operation="SPLIT", is_neg_risk=False, is_yield_bearing=False)
        )[0]
        assert await builder.check_approval_async(step) is False

    @pytest.mark.asyncio
    async def test_erc1155_uses_operator_approval(self, builder_with_signer: OrderBuilder):
        builder = builder_with_signer
        builder._contracts = MagicMock()
        ct = MagicMock()
        ct.functions.isApprovedForAll.return_value.call.return_value = True

        step = builder.get_approval_steps(
            ApprovalScope(
                operation="TRADE", is_neg_risk=False, is_yield_bearing=False, side=Side.SELL
            )
        )[0]
        with patch.object(builder, "_resolve_token_contract", return_value=ct):
            assert await builder.check_approval_async(step) is True
        ct.functions.isApprovedForAll.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_approvals_batches(self, builder_with_signer: OrderBuilder):
        builder = builder_with_signer
        builder._contracts = MagicMock()

        steps = builder.get_approval_steps(
            ApprovalScope(operation="TRADE", is_neg_risk=False, is_yield_bearing=False)
        )
        with patch.object(builder, "check_approval_async", AsyncMock(side_effect=[False, True])):
            checks = await builder.check_approvals_async(steps)

        by_id = {c.step.id: c.satisfied for c in checks}
        assert by_id == {
            "ERC1155_APPROVAL:CTF_EXCHANGE": False,
            "ERC20_ALLOWANCE:CTF_EXCHANGE": True,
        }

    def test_raises_without_signer(self, builder: OrderBuilder):
        step = builder.get_approval_steps(
            ApprovalScope(operation="SPLIT", is_neg_risk=False, is_yield_bearing=False)
        )[0]
        with pytest.raises(MissingSignerError):
            builder.check_approval(step)


class TestSetApproval:
    @pytest.mark.asyncio
    async def test_erc20_routes_to_usdt_approve(self, builder_with_signer: OrderBuilder):
        builder = builder_with_signer
        contracts = MagicMock()
        builder._contracts = contracts
        builder._handle_transaction_async = AsyncMock(  # type: ignore[method-assign]
            return_value=TransactionSuccess(success=True)
        )

        step = builder.get_approval_steps(
            ApprovalScope(operation="SPLIT", is_neg_risk=False, is_yield_bearing=False)
        )[0]
        result = await builder.set_approval_async(step)

        assert result.success is True
        builder._handle_transaction_async.assert_called_once_with(
            contracts.usdt, "approve", ADDRESSES.CONDITIONAL_TOKENS, MAX_UINT256
        )

    @pytest.mark.asyncio
    async def test_erc1155_routes_to_set_approval_for_all(self, builder_with_signer: OrderBuilder):
        builder = builder_with_signer
        builder._contracts = MagicMock()
        builder._handle_transaction_async = AsyncMock(  # type: ignore[method-assign]
            return_value=TransactionSuccess(success=True)
        )
        ct = MagicMock()

        step = builder.get_approval_steps(
            ApprovalScope(
                operation="TRADE", is_neg_risk=False, is_yield_bearing=False, side=Side.SELL
            )
        )[0]
        with patch.object(builder, "_resolve_token_contract", return_value=ct):
            result = await builder.set_approval_async(step)

        assert result.success is True
        builder._handle_transaction_async.assert_called_once_with(
            ct, "setApprovalForAll", ADDRESSES.CTF_EXCHANGE, True
        )

    @pytest.mark.asyncio
    async def test_revocation_via_approved_false(self, builder_with_signer: OrderBuilder):
        builder = builder_with_signer
        builder._contracts = MagicMock()
        builder._handle_transaction_async = AsyncMock(  # type: ignore[method-assign]
            return_value=TransactionSuccess(success=True)
        )
        ct = MagicMock()

        step = builder.get_approval_steps(
            ApprovalScope(
                operation="TRADE", is_neg_risk=False, is_yield_bearing=False, side=Side.SELL
            )
        )[0]
        with patch.object(builder, "_resolve_token_contract", return_value=ct):
            await builder.set_approval_async(step, approved=False)

        builder._handle_transaction_async.assert_called_once_with(
            ct, "setApprovalForAll", ADDRESSES.CTF_EXCHANGE, False
        )

    @pytest.mark.asyncio
    async def test_erc20_revokes_when_approved_false(self, builder_with_signer: OrderBuilder):
        builder = builder_with_signer
        contracts = MagicMock()
        builder._contracts = contracts
        builder._handle_transaction_async = AsyncMock(  # type: ignore[method-assign]
            return_value=TransactionSuccess(success=True)
        )

        step = builder.get_approval_steps(
            ApprovalScope(operation="SPLIT", is_neg_risk=False, is_yield_bearing=False)
        )[0]
        await builder.set_approval_async(step, approved=False)

        # approved=False revokes by setting the ERC-20 allowance to 0 (not MAX_UINT256).
        builder._handle_transaction_async.assert_called_once_with(
            contracts.usdt, "approve", ADDRESSES.CONDITIONAL_TOKENS, 0
        )

    def test_raises_without_signer(self, builder: OrderBuilder):
        step = builder.get_approval_steps(
            ApprovalScope(operation="SPLIT", is_neg_risk=False, is_yield_bearing=False)
        )[0]
        with pytest.raises(MissingSignerError):
            builder.set_approval(step)


class TestRunApprovals:
    @pytest.mark.asyncio
    async def test_skips_satisfied_submits_rest_and_reports(self, builder: OrderBuilder):
        # First step already approved, second needs submitting.
        builder.check_approval_async = AsyncMock(side_effect=[True, False])  # type: ignore[method-assign]
        builder.set_approval_async = AsyncMock(  # type: ignore[method-assign]
            return_value=TransactionSuccess(success=True)
        )

        progress: list[str] = []

        def on_progress(p: ApprovalProgress) -> None:
            progress.append(f"{p.step.id}:{p.status}")

        report = await builder.run_approvals_async(
            builder.get_approval_steps(
                ApprovalScope(operation="TRADE", is_neg_risk=False, is_yield_bearing=False)
            ),
            on_progress=on_progress,
        )

        assert builder.set_approval_async.call_count == 1
        assert report.success is True
        assert [(r.step.id, r.status) for r in report.steps] == [
            ("ERC1155_APPROVAL:CTF_EXCHANGE", "skipped"),
            ("ERC20_ALLOWANCE:CTF_EXCHANGE", "confirmed"),
        ]
        assert progress == [
            "ERC1155_APPROVAL:CTF_EXCHANGE:checking",
            "ERC1155_APPROVAL:CTF_EXCHANGE:skipped",
            "ERC20_ALLOWANCE:CTF_EXCHANGE:checking",
            "ERC20_ALLOWANCE:CTF_EXCHANGE:submitting",
            "ERC20_ALLOWANCE:CTF_EXCHANGE:confirmed",
        ]

    @pytest.mark.asyncio
    async def test_stops_on_first_failure_by_default(self, builder: OrderBuilder):
        builder.check_approval_async = AsyncMock(return_value=False)  # type: ignore[method-assign]
        builder.set_approval_async = AsyncMock(  # type: ignore[method-assign]
            return_value=TransactionFail(success=False)
        )

        report = await builder.run_approvals_async(
            builder.get_approval_steps(
                ApprovalScope(operation="TRADE", is_neg_risk=True, is_yield_bearing=False)
            )
        )

        assert report.success is False
        assert builder.set_approval_async.call_count == 1
        assert len(report.steps) == 1
        assert report.steps[0].status == "failed"

    @pytest.mark.asyncio
    async def test_continues_when_stop_on_error_false(self, builder: OrderBuilder):
        builder.check_approval_async = AsyncMock(return_value=False)  # type: ignore[method-assign]
        builder.set_approval_async = AsyncMock(  # type: ignore[method-assign]
            return_value=TransactionFail(success=False)
        )

        report = await builder.run_approvals_async(
            builder.get_approval_steps(
                ApprovalScope(operation="TRADE", is_neg_risk=True, is_yield_bearing=False)
            ),
            stop_on_error=False,
        )

        assert report.success is False
        assert builder.set_approval_async.call_count == 3
        assert len(report.steps) == 3

    @pytest.mark.asyncio
    async def test_empty_plan_returns_success(self, builder: OrderBuilder):
        builder.check_approval_async = AsyncMock()  # type: ignore[method-assign]

        report = await builder.run_approvals_async(
            builder.get_approval_steps(
                ApprovalScope(operation="MERGE", is_neg_risk=False, is_yield_bearing=False)
            )
        )

        assert report.success is True
        assert report.steps == []
        builder.check_approval_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_dedupes_steps_by_id(self, builder: OrderBuilder):
        builder.check_approval_async = AsyncMock(return_value=False)  # type: ignore[method-assign]
        builder.set_approval_async = AsyncMock(  # type: ignore[method-assign]
            return_value=TransactionSuccess(success=True)
        )

        steps = builder.get_approval_steps(
            ApprovalScope(operation="TRADE", is_neg_risk=False, is_yield_bearing=False)
        )
        report = await builder.run_approvals_async([*steps, *steps])  # duplicated on purpose

        assert builder.set_approval_async.call_count == 2  # 2 unique ids, not 4
        assert len(report.steps) == 2

    @pytest.mark.asyncio
    async def test_no_checking_status_when_skip_satisfied_false(self, builder: OrderBuilder):
        check = AsyncMock()
        builder.check_approval_async = check  # type: ignore[method-assign]
        builder.set_approval_async = AsyncMock(  # type: ignore[method-assign]
            return_value=TransactionSuccess(success=True)
        )

        statuses: list[str] = []
        steps = builder.get_approval_steps(
            ApprovalScope(operation="TRADE", is_neg_risk=False, is_yield_bearing=False)
        )
        report = await builder.run_approvals_async(
            steps, skip_satisfied=False, on_progress=lambda p: statuses.append(p.status)
        )

        check.assert_not_called()
        assert "checking" not in statuses
        assert builder.set_approval_async.call_count == 2  # nothing skipped, both submitted
        assert report.success is True

    @pytest.mark.asyncio
    async def test_failed_precheck_still_submits(self, builder: OrderBuilder):
        # A transient RPC failure during the pre-check must not abort the run.
        builder.check_approval_async = AsyncMock(side_effect=Exception("rpc down"))  # type: ignore[method-assign]
        builder.set_approval_async = AsyncMock(  # type: ignore[method-assign]
            return_value=TransactionSuccess(success=True)
        )

        steps = builder.get_approval_steps(
            ApprovalScope(operation="TRADE", is_neg_risk=False, is_yield_bearing=False)
        )
        report = await builder.run_approvals_async(steps)  # skip_satisfied defaults to True

        assert builder.set_approval_async.call_count == 2  # did not abort on the read failure
        assert report.success is True

    def test_sync_wrapper(self, builder: OrderBuilder):
        builder.check_approval_async = AsyncMock(return_value=False)  # type: ignore[method-assign]
        builder.set_approval_async = AsyncMock(  # type: ignore[method-assign]
            return_value=TransactionSuccess(success=True)
        )

        report = builder.run_approvals(
            builder.get_approval_steps(
                ApprovalScope(
                    operation="TRADE", is_neg_risk=False, is_yield_bearing=False, side=Side.BUY
                )
            )
        )
        assert report.success is True
        assert len(report.steps) == 1


class TestGetAllApprovalSteps:
    def test_single_track_full_deduped_set(self, builder: OrderBuilder):
        steps = builder.get_all_approval_steps(is_yield_bearing=False)
        assert ids(steps) == [
            "ERC1155_APPROVAL:CTF_EXCHANGE",
            "ERC20_ALLOWANCE:CTF_EXCHANGE",
            "ERC20_ALLOWANCE:CONDITIONAL_TOKENS",
            "ERC1155_APPROVAL:NEG_RISK_CTF_EXCHANGE",
            "ERC1155_APPROVAL:NEG_RISK_ADAPTER",
            "ERC20_ALLOWANCE:NEG_RISK_CTF_EXCHANGE",
            "ERC20_ALLOWANCE:NEG_RISK_ADAPTER",
        ]
        # The neg-risk adapter approval is shared by several operations but appears once.
        assert ids(steps).count("ERC1155_APPROVAL:NEG_RISK_ADAPTER") == 1

    def test_both_tracks_no_duplicates(self, builder: OrderBuilder):
        steps = builder.get_all_approval_steps()
        all_ids = ids(steps)
        assert len(all_ids) == 14  # 7 per track
        assert len(set(all_ids)) == len(all_ids)  # no duplicates
        assert all_ids[:7] == ids(builder.get_all_approval_steps(is_yield_bearing=False))
        assert all_ids[7:] == ids(builder.get_all_approval_steps(is_yield_bearing=True))


class TestRunAllSteps:
    @pytest.mark.asyncio
    async def test_runs_full_set(self, builder: OrderBuilder):
        builder.check_approval_async = AsyncMock(return_value=False)  # type: ignore[method-assign]
        builder.set_approval_async = AsyncMock(  # type: ignore[method-assign]
            return_value=TransactionSuccess(success=True)
        )

        report = await builder.run_approvals_async(
            builder.get_all_approval_steps(is_yield_bearing=False)
        )

        assert report.success is True
        assert builder.set_approval_async.call_count == 7
        assert len(report.steps) == 7
