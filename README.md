# Predict SDK for Python

A Python SDK to help developers interface with the [Predict.fun](https://predict.fun/) protocol.

## Installation

```bash
pip install predict-sdk
```

## Quick Start

```python
from predict_sdk import (
    OrderBuilder,
    ChainId,
    Side,
    BuildOrderInput,
    LimitHelperInput,
)

# Create an OrderBuilder without a signer (read-only)
builder = OrderBuilder.make(ChainId.BNB_MAINNET)

# Calculate order amounts for a LIMIT order
amounts = builder.get_limit_order_amounts(
    LimitHelperInput(
        side=Side.BUY,
        price_per_share_wei=500000000000000000,  # 0.5 USDT per share
        quantity_wei=10000000000000000000,  # 10 shares
    )
)

print(f"Maker Amount: {amounts.maker_amount}")
print(f"Taker Amount: {amounts.taker_amount}")

# Build an order
order = builder.build_order(
    "LIMIT",
    BuildOrderInput(
        side=Side.BUY,
        token_id="YOUR_TOKEN_ID",
        maker_amount=str(amounts.maker_amount),
        taker_amount=str(amounts.taker_amount),
        fee_rate_bps=100,  # Get from GET /markets endpoint
    ),
)
```

## With a Signer

```python
from eth_account import Account
from predict_sdk import OrderBuilder, ChainId

# Create from private key
private_key = "0x..."  # Your private key
signer = Account.from_key(private_key)

# Or pass private key directly
builder = OrderBuilder.make(ChainId.BNB_MAINNET, private_key)

# Now you can sign orders and interact with contracts
typed_data = builder.build_typed_data(
    order,
    is_neg_risk=False,
    is_yield_bearing=False,
)

signed_order = builder.sign_typed_data_order(typed_data)
print(f"Signature: {signed_order.signature}")
```

## Setting Approvals

Before trading, you need to set approvals for the exchange contracts. A single
`set_approvals()` call covers every market type (standard and yield-bearing). It is
idempotent: approvals already in place are skipped, so re-running it only sends the
transactions that are still missing.

```python
# Set approvals for ALL market types (standard + yield-bearing) in one call
result = builder.set_approvals()

if result.success:
    print("All approvals set successfully!")
else:
    print("Some approvals failed")
    for tx in result.transactions:
        if not tx.success:
            print(f"Failed: {tx.cause}")

# Optional: limit the run to a single track
builder.set_approvals(is_yield_bearing=False)  # standard markets only
builder.set_approvals(is_yield_bearing=True)   # yield-bearing markets only

# Or set individual approvals
builder.set_ctf_exchange_approval(is_neg_risk=False, is_yield_bearing=False)
builder.set_ctf_exchange_allowance(is_neg_risk=False, is_yield_bearing=False)
```

## Scoped Approvals (per-operation)

`set_approvals()` sets every approval for every market type in a single call. For most apps the scoped-approvals API is the better fit, and is the recommended approach whenever you build an approval UI. It returns only the approvals a given operation needs, as a list of plain, self-describing steps you can render as a checklist, pre-check, run with live progress, and gate on user confirmation.

The mental model is simple: **everything produces steps, and one runner runs steps.**

1. Describe what the user is about to do with an `ApprovalScope` (`operation`, plus `is_neg_risk` / `is_yield_bearing`, and an optional `side` to narrow a `TRADE`).
2. Turn it into an ordered list of `ApprovalStep`s with `get_approval_steps` (one operation) or `get_all_approval_steps` (everything). Both are pure (no signer, no network access), so you can render the checklist before the wallet is connected.
3. Run the steps with `run_approvals`, or drive them yourself with `check_approvals` / `set_approval`.

`is_neg_risk` and `is_yield_bearing` describe the market and can be fetched from the `GET /markets` (or `GET /categories`) endpoint.

`get_approval_steps` returns the minimal, ordered set for the operation. The labels below are the SDK's default copy (they match the Predict web app). Operations that need no approval return an empty list.

A `TRADE` scope covers both order directions by default. Pass `side=Side.BUY` for just the collateral allowance, or `side=Side.SELL` for just the ERC-1155 approval. `CONVERT` is neg-risk only and raises `InvalidApprovalOperationError` for a standard market.

### The `ApprovalStep` shape

Each step is a plain dataclass, safe to render and serialize:

```python
ApprovalStep(
    id="ERC1155_APPROVAL:CTF_EXCHANGE",  # stable identifier: "{type}:{spender_key}"
    type="ERC1155_APPROVAL",             # or "ERC20_ALLOWANCE"
    spender="0x8BC0...B689",             # the contract being granted permission
    token="0x22DA...d244",               # the token contract (conditional tokens for ERC-1155, USDT for ERC-20)
    label="Approve Exchange",            # default copy
    description="Allows you to interact with the exchange.",
)
```

The `label`/`description` are sensible English defaults. For custom wording or i18n, key your own copy off the stable `id` and ignore them.

### Build the steps

You'll typically build the steps from the same signer-backed `OrderBuilder` you use to check and run them.

```python
from predict_sdk import OrderBuilder, ChainId, Side, ApprovalScope

builder = OrderBuilder.make(ChainId.BNB_MAINNET, private_key)

# For a single operation on a market:
steps = builder.get_approval_steps(
    ApprovalScope(
        operation="TRADE",
        is_neg_risk=True,
        is_yield_bearing=False,
        # side=Side.BUY,  # optional TRADE narrowing
    )
)

# For onboarding, every approval across both market types (and, by default, both tracks):
all_steps = builder.get_all_approval_steps()  # or is_yield_bearing=False to limit to one track
```

`get_approval_steps` and `get_all_approval_steps` don't touch the chain, so you _can_ also call them on a signer-less builder (`OrderBuilder.make(ChainId.BNB_MAINNET)`) to render the checklist before the wallet is connected. You'll need a signer for everything after (`check_approvals`, `set_approval`, `run_approvals`).

### Run them with progress reporting

`run_approvals(steps, ...)` runs the steps in order, deduplicating by `id` (so you can safely pass a union of scopes or a curated subset). Options:

- `skip_satisfied` (default `True`): pre-check each step on-chain and skip the ones already in place.
- `stop_on_error` (default `True`): stop after the first failed step.
- `on_progress`: a callback invoked as each step transitions, receiving an `ApprovalProgress(step, status, transaction)`.

```python
builder = OrderBuilder.make(ChainId.BNB_MAINNET, private_key)

steps = builder.get_approval_steps(
    ApprovalScope(operation="TRADE", is_neg_risk=True, is_yield_bearing=False)
)

report = builder.run_approvals(
    steps,
    skip_satisfied=True,     # default
    stop_on_error=True,      # default
    on_progress=lambda p: update_ui(p.step.id, p.status),
)
```

`on_progress` reports each step through this lifecycle:

| Status       | Meaning                                                                     |
| ------------ | --------------------------------------------------------------------------- |
| `checking`   | reading on-chain whether it's already approved (only when `skip_satisfied`) |
| `skipped`    | already in place, nothing sent                                              |
| `submitting` | transaction sent, awaiting confirmation                                     |
| `confirmed`  | the transaction succeeded                                                   |
| `failed`     | the transaction reverted or failed                                          |

The returned `ApprovalRunReport` has `success` and `steps`, where each entry is an `ApprovalStepResult(step, status, transaction)` with `status` one of `"skipped"`, `"confirmed"`, or `"failed"`:

```python
if not report.success:
    failed = [s.step.id for s in report.steps if s.status == "failed"]
    raise RuntimeError(f"Approvals failed: {failed}")

# Async variant:
# report = await builder.run_approvals_async(steps, on_progress=...)
```

### Render a live checklist (the typical UI flow)

This is the flow behind an in-app "Approvals" modal: render the steps, mark the ones already done, then run the rest with live updates.

```python
builder = OrderBuilder.make(ChainId.BNB_MAINNET, private_key)

# 1. Build the plan.
steps = builder.get_approval_steps(
    ApprovalScope(operation="TRADE", is_neg_risk=is_neg_risk, is_yield_bearing=is_yield_bearing)
)

# 2. Render the checklist, marking which are already satisfied.
for check in builder.check_approvals(steps):
    add_row(check.step.id, check.step.label, check.step.description,
            "done" if check.satisfied else "pending")

# 3. Run the remaining steps, updating each row as it progresses.
report = builder.run_approvals(
    steps,
    on_progress=lambda p: set_row_status(p.step.id, p.status),
)
```

For first-time onboarding, swap `get_approval_steps(scope)` for `get_all_approval_steps()` to approve everything the protocol could need in one pass. That is the per-step, progress-reportable equivalent of `set_approvals()` (and a slight superset, since it also includes the split allowances).

### Or drive each step yourself

For full control (e.g. gating each step on a user confirmation), use the per-step primitives.

```python
builder = OrderBuilder.make(ChainId.BNB_MAINNET, private_key)
steps = builder.get_approval_steps(
    ApprovalScope(operation="SPLIT", is_neg_risk=False, is_yield_bearing=False)
)

# Batched pre-check to mark which steps are already done.
for check in builder.check_approvals(steps):
    if check.satisfied:
        continue  # already approved
    # set_approval is a raw send: ERC-20 defaults to MAX_UINT256, ERC-1155 to approved=True.
    # Pass approved=False to revoke, or amount=... to cap an allowance.
    result = builder.set_approval(check.step)
    if not result.success:
        break  # you decide whether to continue
```

### Notes

- **Compose freely.** Union step lists to cover several operations at once, e.g. to make a market both trade-ready and splittable: `run_approvals([*trade_steps, *split_steps])` (duplicates are removed automatically).
- **Predict accounts** (smart wallets) are supported transparently: every step routes through `Kernel.execute` when a `predict_account` is configured.
- **Signer requirements.** `get_approval_steps` and `get_all_approval_steps` are pure and need no signer. `check_approval` / `check_approvals`, `set_approval`, and `run_approvals` require one and raise `MissingSignerError` otherwise.
- **Async.** Every contract-touching method has an `_async` variant (`run_approvals_async`, `check_approvals_async`, `set_approval_async`, ...). The step getters are sync (pure).
- **`set_approvals()` still exists** for the fire-and-forget "approve everything" case where you don't need per-step control or reporting.

## Using Predict Accounts (Smart Wallets)

```python
from predict_sdk import OrderBuilder, ChainId, OrderBuilderOptions

builder = OrderBuilder.make(
    ChainId.BNB_MAINNET,
    private_key,  # Must be the Privy exported wallet
    OrderBuilderOptions(
        predict_account="0x...",  # Your Predict account address
    ),
)
```

## Signing Messages with Predict Accounts

When using a Predict account (smart wallet), you can sign arbitrary messages that can be verified on-chain:

```python
from predict_sdk import OrderBuilder, ChainId, OrderBuilderOptions

builder = OrderBuilder.make(
    ChainId.BNB_MAINNET,
    private_key,
    OrderBuilderOptions(predict_account="0x..."),
)

# Sign a string message
signature = builder.sign_predict_account_message("Hello, world!")

# Or sign a pre-computed hash (useful for EIP-712 typed data)
signature = builder.sign_predict_account_message({"raw": "0x1234..."})

# Async version available
signature = await builder.sign_predict_account_message_async("Hello, world!")
```

The signature is formatted for Kernel smart wallet verification, including the ECDSA validator address prefix.

## Market Orders

```python
from predict_sdk import MarketHelperInput, MarketHelperValueInput

# Market order by quantity (number of shares)
amounts = builder.get_market_order_amounts(
    MarketHelperInput(
        side=Side.BUY,
        quantity_wei=10000000000000000000,  # 10 shares
    ),
    orderbook,  # From GET /orderbook/{marketId}
)

# Market BUY by value (total USDT to spend)
amounts = builder.get_market_order_amounts(
    MarketHelperValueInput(
        side=Side.BUY,
        value_wei=5000000000000000000,  # 5 USDT
    ),
    orderbook,
)

print(f"Maker Amount: {amounts.maker_amount}")
print(f"Taker Amount: {amounts.taker_amount}")
print(f"Amount: {amounts.amount}")
print(f"Price Per Share: {amounts.price_per_share}")
print(f"Slippage Applied: {amounts.slippage_bps} bps")
print(f"Is Min Amount Out: {amounts.is_min_amount_out}")
```

### Slippage

By default, no additional slippage is applied to the order maker/taker amounts. You can specify a slippage tolerance in basis points (1 bps = 0.01%) to adjust the amounts:

- **BUY orders**: slippage deflates the `taker_amount` (minimum shares out), keeping the USD commitment minimal so users can spend their full wallet balance. Requires `is_min_amount_out=True`.
- **SELL orders**: slippage decreases the `taker_amount` (you're willing to receive less)

```python
# BUY with slippage
amounts = builder.get_market_order_amounts(
    MarketHelperInput(
        side=Side.BUY,
        quantity_wei=10000000000000000000,  # 10 shares
        slippage_bps=100,  # 1% slippage tolerance
        is_min_amount_out=True,
    ),
    orderbook,
)

print(f"Maker Amount (cost): {amounts.maker_amount}")
print(f"Taker Amount (min shares out): {amounts.taker_amount}")
print(f"Amount (actual shares): {amounts.amount}")
print(f"Slippage: {amounts.slippage_bps} bps")  # 100

# Value-based BUY with slippage
amounts = builder.get_market_order_amounts(
    MarketHelperValueInput(
        side=Side.BUY,
        value_wei=5000000000000000000,  # 5 USDT
        slippage_bps=50,  # 0.5% slippage tolerance
        is_min_amount_out=True,
    ),
    orderbook,
)
```

**Note:** You must forward `is_min_amount_out=True`, the `amount` field, and `slippageBps` in your REST API request body for slippage to be applied.

## Redeeming Positions

```python
# For standard markets
result = builder.redeem_positions(
    condition_id="0x...",
    index_set=1,  # 1 or 2
    is_neg_risk=False,
    is_yield_bearing=False,
)

# For NegRisk (winner-takes-all) markets
result = builder.redeem_positions(
    condition_id="0x...",
    index_set=1,
    amount=1000000000000000000,  # Required for NegRisk
    is_neg_risk=True,
    is_yield_bearing=False,
)
```

## Merging Positions

Merge both outcome tokens back into collateral (USDT). Useful when holding equal amounts of both YES and NO positions.

```python
# For standard markets
result = builder.merge_positions(
    condition_id="0x...",
    amount=1000000000000000000,  # Amount to merge (in wei)
    is_neg_risk=False,
    is_yield_bearing=False,
)

# For NegRisk (winner-takes-all) markets
result = builder.merge_positions(
    condition_id="0x...",
    amount=1000000000000000000,
    is_neg_risk=True,
    is_yield_bearing=False,
)
```

## Canceling Orders

```python
from predict_sdk import CancelOrdersOptions

result = builder.cancel_orders(
    orders=[order1, order2],
    options=CancelOrdersOptions(
        is_neg_risk=False,
        is_yield_bearing=False,
    ),
)
```

## Checking Balance

```python
balance = builder.balance_of("USDT")
print(f"USDT Balance: {balance}")
```

## Async Support

All contract interaction methods have async versions:

```python
import asyncio

async def main():
    balance = await builder.balance_of_async("USDT")
    result = await builder.set_approvals_async()

asyncio.run(main())
```

## API Reference

### OrderBuilder

The main class for building and signing orders.

#### Factory Method

```python
OrderBuilder.make(
    chain_id: ChainId,
    signer: LocalAccount | str | None = None,
    options: OrderBuilderOptions | None = None,
) -> OrderBuilder
```

#### Order Building Methods

- `get_limit_order_amounts(data: LimitHelperInput) -> OrderAmounts`
- `get_market_order_amounts(data: MarketHelperInput | MarketHelperValueInput, book: Book) -> OrderAmounts`
- `build_order(strategy: "MARKET" | "LIMIT", data: BuildOrderInput) -> Order`
- `build_typed_data(order: Order, *, is_neg_risk: bool, is_yield_bearing: bool) -> EIP712TypedData`
- `build_typed_data_hash(typed_data: EIP712TypedData) -> str`
- `sign_typed_data_order(typed_data: EIP712TypedData) -> SignedOrder`
- `sign_predict_account_message(message: str | dict) -> str` - Sign a message for a Predict account
- `sign_predict_account_message_async(message: str | dict) -> str` - Async version

#### Approval Methods

- `set_approvals(*, is_yield_bearing: bool | None = None) -> SetApprovalsResult` (blanket; `None` approves both tracks)
- `set_ctf_exchange_approval(*, is_neg_risk: bool, is_yield_bearing: bool, approved: bool = True) -> TransactionResult`
- `set_neg_risk_adapter_approval(*, is_yield_bearing: bool, approved: bool = True) -> TransactionResult`
- `set_ctf_exchange_allowance(*, is_neg_risk: bool, is_yield_bearing: bool, amount: int = MAX_UINT256) -> TransactionResult`

#### Scoped Approvals

Each contract-touching method also has an `_async` variant (`check_approvals_async`, `set_approval_async`, `run_approvals_async`, ...). The step getters are pure (sync, no signer).

- `get_approval_steps(scope: ApprovalScope) -> list[ApprovalStep]`
- `get_all_approval_steps(*, is_yield_bearing: bool | None = None) -> list[ApprovalStep]`
- `check_approval(step: ApprovalStep) -> bool`
- `check_approvals(steps: list[ApprovalStep]) -> list[ApprovalCheck]`
- `set_approval(step: ApprovalStep, *, approved: bool = True, amount: int = MAX_UINT256) -> TransactionResult`
- `run_approvals(steps: list[ApprovalStep], *, skip_satisfied: bool = True, stop_on_error: bool = True, on_progress: Callable[[ApprovalProgress], None] | None = None) -> ApprovalRunReport`

#### Position Management

- `balance_of(token: "USDT" = "USDT", address: str | None = None) -> int`
- `redeem_positions(condition_id: str, index_set: 1 | 2, amount: int | None = None, *, is_neg_risk: bool, is_yield_bearing: bool) -> TransactionResult`
- `merge_positions(condition_id: str, amount: int, *, is_neg_risk: bool, is_yield_bearing: bool) -> TransactionResult`

#### Order Cancellation

- `cancel_orders(orders: list[Order], options: CancelOrdersOptions) -> TransactionResult`
- `validate_token_ids(token_ids: list[int], *, is_neg_risk: bool, is_yield_bearing: bool) -> bool`

### Types

```python
from predict_sdk import (
    ChainId,       # BNB_MAINNET (56), BNB_TESTNET (97)
    Side,          # BUY (0), SELL (1)
    SignatureType, # Only supports EOA (0)
    Order,
    SignedOrder,
    BuildOrderInput,
    OrderAmounts,
    LimitHelperInput,
    MarketHelperInput,
    MarketHelperValueInput,
    Book,
    EIP712TypedData,
    TransactionResult,
    SetApprovalsResult,
    CancelOrdersOptions,
    OrderBuilderOptions,
    # Scoped approvals
    ApprovalScope,        # operation + is_neg_risk + is_yield_bearing + side?
    ApprovalStep,         # id, type, spender, token, label, description
    ApprovalCheck,        # step + satisfied
    ApprovalProgress,     # step + status + transaction (on_progress payload)
    ApprovalStepResult,   # step + status + transaction (report entry)
    ApprovalRunReport,    # success + steps
)
```

## Requirements

- Python >= 3.10
- web3.py >= 6.0.0

## License

MIT
