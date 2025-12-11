"""Type definitions for the Predict SDK."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from web3.types import TxReceipt

from predict_sdk.constants import Side, SignatureType

# Type aliases
Address = str
BigIntString = str
Currency = Literal["USDT"]
OrderStrategy = Literal["MARKET", "LIMIT"]
QuoteType = bool  # True = Ask, False = Bid
LogLevel = Literal["ERROR", "WARN", "INFO", "DEBUG"]


@dataclass
class LimitHelperInput:
    """Input for calculating limit order amounts."""

    side: Side
    price_per_share_wei: int
    quantity_wei: int


@dataclass
class MarketHelperInput:
    """
    Input for calculating market order amounts by quantity.

    The quantity of shares you would like to trade. This can be used for
    SELL or BUY orders, however is most commonly used for SELL orders.
    """

    side: Side
    quantity_wei: int


@dataclass
class MarketHelperValueInput:
    """
    Input for calculating market order amounts by value.

    The total maximum value to spend on the order. This is only used for BUY orders.
    """

    side: Literal[Side.BUY]
    value_wei: int


@dataclass
class ProcessedBookAmounts:
    """Processed orderbook amounts."""

    quantity_wei: int
    price_wei: int
    last_price_wei: int


@dataclass
class OrderAmounts:
    """Calculated order amounts."""

    last_price: int
    price_per_share: int
    maker_amount: int
    taker_amount: int


@dataclass
class Order:
    """
    Represents a trading order.

    Attributes:
        salt: A unique salt to ensure entropy
        maker: The maker of the order, e.g. the order's signer
        signer: The signer of the order
        taker: The address of the order taker. Zero address indicates a public order
        token_id: The token ID of the CTF ERC-1155 asset to be bought or sold
        maker_amount: For BUY: total collateral offered. For SELL: total CTF assets offered
        taker_amount: For BUY: total CTF assets to receive. For SELL: total collateral to receive
        expiration: The timestamp in seconds after which the order is expired
        nonce: The nonce used for on-chain cancellations
        fee_rate_bps: The fee rate, in basis points
        side: The side of the order, BUY (Bid) or SELL (Ask)
        signature_type: Signature type used by the Order (EOA also supports EIP-1271)
    """

    salt: BigIntString
    maker: Address
    signer: Address
    taker: Address
    token_id: BigIntString
    maker_amount: BigIntString
    taker_amount: BigIntString
    expiration: BigIntString
    nonce: BigIntString
    fee_rate_bps: BigIntString
    side: Side
    signature_type: SignatureType


@dataclass
class OrderWithHash(Order):
    """Order with computed hash."""

    hash: str = ""


@dataclass
class SignedOrder(Order):
    """Signed order with signature."""

    signature: str = ""
    hash: str | None = None


@dataclass
class BuildOrderInput:
    """
    Input for building an order.

    Attributes:
        side: The side of the order (BUY or SELL)
        token_id: The token ID of the CTF ERC-1155 asset
        maker_amount: The maker amount
        taker_amount: The taker amount
        fee_rate_bps: The current fee rate (fetch via GET /markets endpoint)
        signer: Optional signer address
        nonce: Optional nonce for cancellations
        salt: Optional salt for entropy
        maker: Optional maker address
        taker: Optional taker address
        signature_type: Optional signature type
        expires_at: Optional expiration datetime
    """

    side: Side
    token_id: str | int
    maker_amount: str | int
    taker_amount: str | int
    fee_rate_bps: str | int
    signer: Address | None = None
    nonce: str | int | None = None
    salt: str | int | None = None
    maker: Address | None = None
    taker: Address | None = None
    signature_type: SignatureType | None = None
    expires_at: datetime | None = None


# EIP-712 Types
EIP712ObjectValue = str | int | dict[str, Any]
EIP712Object = dict[str, EIP712ObjectValue]
EIP712Types = dict[str, list[dict[str, str]]]


@dataclass
class EIP712TypedData:
    """EIP-712 typed data structure."""

    types: EIP712Types
    domain: EIP712Object
    message: EIP712Object
    primary_type: str


# Orderbook Types
DepthLevel = tuple[float, float]  # (price, quantity)


@dataclass
class Book:
    """Orderbook data."""

    market_id: int
    update_timestamp_ms: int
    asks: list[DepthLevel]
    bids: list[DepthLevel]


# Transaction Result Types
@dataclass
class TransactionSuccess:
    """Successful transaction result."""

    success: Literal[True] = True
    receipt: TxReceipt | None = None


@dataclass
class TransactionFail:
    """Failed transaction result."""

    success: Literal[False] = False
    cause: Exception | None = None
    receipt: TxReceipt | None = None


TransactionResult = TransactionSuccess | TransactionFail


@dataclass
class SetApprovalsResult:
    """Result of setting all approvals."""

    success: bool
    transactions: list[TransactionResult] = field(default_factory=list)


@dataclass
class CancelOrdersOptions:
    """Options for canceling orders."""

    is_yield_bearing: bool
    is_neg_risk: bool
    with_validation: bool = True


@dataclass
class OrderBuilderOptions:
    """
    Configuration options for OrderBuilder.

    Attributes:
        precision: The number of decimals supported. Default is 18 (for wei).
        predict_account: When defined, the OrderBuilder signer must be the Privy
            exported wallet from the account's settings.
        generate_salt: Optional custom salt generator function.
        log_level: Logging level for the OrderBuilder.
    """

    precision: int = 18
    predict_account: Address | None = None
    generate_salt: Callable[[], str] | None = None
    log_level: LogLevel = "INFO"
