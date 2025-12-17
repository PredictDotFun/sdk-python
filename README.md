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

Before trading, you need to set approvals for the exchange contracts:

```python
# Set all approvals at once
result = builder.set_approvals(is_yield_bearing=False)

if result.success:
    print("All approvals set successfully!")
else:
    print("Some approvals failed")
    for tx in result.transactions:
        if not tx.success:
            print(f"Failed: {tx.cause}")

# Or set individual approvals
builder.set_ctf_exchange_approval(is_neg_risk=False, is_yield_bearing=False)
builder.set_ctf_exchange_allowance(is_neg_risk=False, is_yield_bearing=False)
```

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
```

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
    result = await builder.set_approvals_async(is_yield_bearing=False)

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

- `set_approvals(*, is_yield_bearing: bool = False) -> SetApprovalsResult`
- `set_ctf_exchange_approval(*, is_neg_risk: bool, is_yield_bearing: bool, approved: bool = True) -> TransactionResult`
- `set_neg_risk_adapter_approval(*, is_yield_bearing: bool, approved: bool = True) -> TransactionResult`
- `set_ctf_exchange_allowance(*, is_neg_risk: bool, is_yield_bearing: bool, amount: int = MAX_UINT256) -> TransactionResult`

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
)
```

## Requirements

- Python >= 3.10
- web3.py >= 6.0.0

## License

MIT
