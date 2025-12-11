"""Custom exceptions for the Predict SDK."""

from __future__ import annotations


class PredictSDKError(Exception):
    """Base exception for all SDK errors."""

    pass


class MissingSignerError(PredictSDKError):
    """Raised when a signer is required but not provided."""

    def __init__(self) -> None:
        super().__init__("A signer is required to sign the order")


class InvalidQuantityError(PredictSDKError):
    """Raised when the quantity or price is invalid."""

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or "Invalid quantityWei. Must be greater than 1e16.")


class InvalidExpirationError(PredictSDKError):
    """Raised when the expiration is invalid."""

    def __init__(self) -> None:
        super().__init__("Invalid expiration. Must be in the future.")


class FailedOrderSignError(PredictSDKError):
    """Raised when signing an order fails."""

    cause: Exception | None

    def __init__(self, cause: Exception | None = None) -> None:
        message = "Failed to EIP-712 sign the order via signTypedData"
        if cause:
            message += f": {cause}"
        super().__init__(message)
        self.cause = cause


class FailedTypedDataEncoderError(PredictSDKError):
    """Raised when encoding typed data fails."""

    cause: Exception | None

    def __init__(self, cause: Exception | None = None) -> None:
        message = "Failed to hash the order's typed data"
        if cause:
            message += f": {cause}"
        super().__init__(message)
        self.cause = cause


class InvalidNegRiskConfig(PredictSDKError):
    """Raised when NegRisk configuration is invalid."""

    def __init__(self) -> None:
        super().__init__(
            "The token ID of one or more orders is not registered in the selected contract. "
            "Use `cancel_orders` with the appropriate `is_neg_risk` parameter."
        )


class MakerSignerMismatchError(PredictSDKError):
    """Raised when maker and signer addresses don't match."""

    def __init__(self) -> None:
        super().__init__("The maker and signer must be the same address.")


class InvalidSignerError(PredictSDKError):
    """Raised when the signer is invalid for the Predict account."""

    def __init__(self) -> None:
        super().__init__(
            "The signer is not the owner of the Predict account or you are on the wrong chain. "
            "The signer must be the Privy wallet exported from your account's settings. "
            "See: https://predict.fun/account/settings"
        )
