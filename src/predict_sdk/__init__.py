"""
Predict SDK for Python.

A Python SDK to help developers interface with the Predict.fun protocol.
"""

from __future__ import annotations

from predict_sdk._internal.utils import generate_order_salt

# ABIs
from predict_sdk.abis import (
    CONDITIONAL_TOKENS_ABI,
    CTF_EXCHANGE_ABI,
    ECDSA_VALIDATOR_ABI,
    ERC20_ABI,
    KERNEL_ABI,
    NEG_RISK_ADAPTER_ABI,
    NEG_RISK_CTF_EXCHANGE_ABI,
    YIELD_BEARING_CONDITIONAL_TOKENS_ABI,
)
from predict_sdk.constants import (
    ADDRESSES_BY_CHAIN_ID,
    EIP712_DOMAIN,
    FIVE_MINUTES_SECONDS,
    KERNEL_DOMAIN_BY_CHAIN_ID,
    MAX_SALT,
    MAX_UINT256,
    ORDER_STRUCTURE,
    PROTOCOL_NAME,
    PROTOCOL_VERSION,
    RPC_URLS_BY_CHAIN_ID,
    ZERO_ADDRESS,
    ZERO_HASH,
    Addresses,
    ChainId,
    Side,
    SignatureType,
)
from predict_sdk.errors import (
    FailedOrderSignError,
    FailedTypedDataEncoderError,
    InvalidExpirationError,
    InvalidNegRiskConfig,
    InvalidQuantityError,
    InvalidSignerError,
    MakerSignerMismatchError,
    MissingSignerError,
    PredictSDKError,
)
from predict_sdk.order_builder import OrderBuilder
from predict_sdk.types import (
    Book,
    BuildOrderInput,
    CancelOrdersOptions,
    DepthLevel,
    EIP712TypedData,
    LimitHelperInput,
    LogLevel,
    MarketHelperInput,
    MarketHelperValueInput,
    Order,
    OrderAmounts,
    OrderBuilderOptions,
    SetApprovalsResult,
    SignedOrder,
    TransactionFail,
    TransactionResult,
    TransactionSuccess,
)

__version__ = "0.0.12"

__all__ = [
    # Version
    "__version__",
    # Main class
    "OrderBuilder",
    # Salt generation
    "generate_order_salt",
    # Types
    "Order",
    "SignedOrder",
    "BuildOrderInput",
    "OrderAmounts",
    "LimitHelperInput",
    "MarketHelperInput",
    "MarketHelperValueInput",
    "Book",
    "DepthLevel",
    "EIP712TypedData",
    "TransactionResult",
    "TransactionSuccess",
    "TransactionFail",
    "SetApprovalsResult",
    "CancelOrdersOptions",
    "OrderBuilderOptions",
    "Addresses",
    "LogLevel",
    # Constants
    "ChainId",
    "Side",
    "SignatureType",
    "ADDRESSES_BY_CHAIN_ID",
    "RPC_URLS_BY_CHAIN_ID",
    "KERNEL_DOMAIN_BY_CHAIN_ID",
    "PROTOCOL_NAME",
    "PROTOCOL_VERSION",
    "MAX_SALT",
    "MAX_UINT256",
    "FIVE_MINUTES_SECONDS",
    "EIP712_DOMAIN",
    "ORDER_STRUCTURE",
    "ZERO_ADDRESS",
    "ZERO_HASH",
    # Errors
    "PredictSDKError",
    "MissingSignerError",
    "InvalidQuantityError",
    "InvalidExpirationError",
    "FailedOrderSignError",
    "FailedTypedDataEncoderError",
    "InvalidNegRiskConfig",
    "MakerSignerMismatchError",
    "InvalidSignerError",
    # ABIs
    "CTF_EXCHANGE_ABI",
    "NEG_RISK_CTF_EXCHANGE_ABI",
    "NEG_RISK_ADAPTER_ABI",
    "CONDITIONAL_TOKENS_ABI",
    "YIELD_BEARING_CONDITIONAL_TOKENS_ABI",
    "ERC20_ABI",
    "KERNEL_ABI",
    "ECDSA_VALIDATOR_ABI",
]
