"""Main OrderBuilder class for creating and signing orders."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Coroutine
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal, TypeVar, overload

from eth_account import Account
from eth_account.messages import _hash_eip191_message, encode_defunct, encode_typed_data
from eth_account.signers.local import LocalAccount
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware

from predict_sdk._internal.contracts import (
    Contracts,
    get_conditional_tokens_contract,
    get_exchange_contract,
    get_neg_risk_adapter_contract,
    make_contract,
    make_contracts,
)
from predict_sdk._internal.utils import (
    eip712_wrap_hash,
    float_to_wei,
    generate_order_salt,
    retain_significant_digits,
)
from predict_sdk.abis import KERNEL_ABI
from predict_sdk.constants import (
    ADDRESSES_BY_CHAIN_ID,
    APPROVAL_STEP_COPY,
    EIP712_DOMAIN,
    FIVE_MINUTES_SECONDS,
    KERNEL_DOMAIN_BY_CHAIN_ID,
    MAX_INT256,
    MAX_UINT256,
    ORDER_STRUCTURE,
    PROTOCOL_NAME,
    PROTOCOL_VERSION,
    RPC_URLS_BY_CHAIN_ID,
    SPENDER_ROLE_BY_KEY,
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
    InvalidApprovalOperationError,
    InvalidExpirationError,
    InvalidQuantityError,
    InvalidSignerError,
    MakerSignerMismatchError,
    MissingSignerError,
)
from predict_sdk.logger import Logger
from predict_sdk.types import (
    ApprovalCheck,
    ApprovalOperation,
    ApprovalProgress,
    ApprovalRunReport,
    ApprovalScope,
    ApprovalStep,
    ApprovalStepResult,
    ApprovalStepType,
    Book,
    BuildOrderInput,
    CancelOrdersOptions,
    DepthLevel,
    EIP712TypedData,
    LimitHelperInput,
    MarketHelperInput,
    MarketHelperValueInput,
    Order,
    OrderAmounts,
    OrderBuilderOptions,
    ProcessedBookAmounts,
    SetApprovalsResult,
    SignedOrder,
    TransactionFail,
    TransactionResult,
    TransactionSuccess,
)

if TYPE_CHECKING:
    from web3.contract import Contract

_T = TypeVar("_T")


class OrderBuilder:
    """
    Helper class to build, sign, and manage orders.

    Use the static `make()` method to create instances.
    """

    def __init__(
        self,
        chain_id: ChainId,
        precision: int,
        addresses: Addresses,
        generate_salt_fn: Callable[[], str],
        logger: Logger,
        signer: LocalAccount | None = None,
        predict_account: str | None = None,
        contracts: Contracts | None = None,
        web3: Web3 | None = None,
    ) -> None:
        self._chain_id = chain_id
        self._precision = 10**precision
        self._addresses = addresses
        self._generate_salt = generate_salt_fn
        self._logger = logger
        self._signer = signer
        self._predict_account = predict_account
        self._contracts = contracts
        self._web3 = web3
        self._execution_mode = bytes.fromhex(ZERO_HASH[2:])

    @overload
    @classmethod
    def make(
        cls,
        chain_id: ChainId,
        signer: None = None,
        options: OrderBuilderOptions | None = None,
    ) -> OrderBuilder: ...

    @overload
    @classmethod
    def make(
        cls,
        chain_id: ChainId,
        signer: LocalAccount | str,
        options: OrderBuilderOptions | None = None,
    ) -> OrderBuilder: ...

    @classmethod
    def make(
        cls,
        chain_id: ChainId,
        signer: LocalAccount | str | None = None,
        options: OrderBuilderOptions | None = None,
    ) -> OrderBuilder:
        """
        Factory method to create an OrderBuilder instance.

        Args:
            chain_id: The chain ID for the network.
            signer: Optional signer (LocalAccount or private key string).
            options: Optional configuration options.

        Returns:
            A new OrderBuilder instance.

        Raises:
            InvalidSignerError: If the signer is not the owner of the Predict account.
        """
        opts = options or OrderBuilderOptions()
        addresses = ADDRESSES_BY_CHAIN_ID[chain_id]
        precision = opts.precision
        generate_salt_fn = opts.generate_salt or generate_order_salt
        predict_account = opts.predict_account
        logger = Logger(opts.log_level)

        contracts = None
        web3 = None
        signer_account = None

        if signer is not None:
            # Convert private key to account if needed
            if isinstance(signer, str):
                signer_account = Account.from_key(signer)
            else:
                signer_account = signer

            # Create Web3 instance
            rpc_url = RPC_URLS_BY_CHAIN_ID[chain_id]
            web3 = Web3(Web3.HTTPProvider(rpc_url))

            # Inject POA middleware for BNB chain (required for extraData validation)
            if chain_id in (ChainId.BNB_MAINNET, ChainId.BNB_TESTNET):
                web3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)

            # Create contract instances
            contracts = make_contracts(web3, addresses, signer_account)

            # Validate Predict account ownership if provided
            if predict_account:
                owner = contracts.ecdsa_validator.functions.ecdsaValidatorStorage(
                    predict_account
                ).call()
                if owner != signer_account.address:
                    raise InvalidSignerError()

        return cls(
            chain_id=chain_id,
            precision=precision,
            addresses=addresses,
            generate_salt_fn=generate_salt_fn,
            logger=logger,
            signer=signer_account,
            predict_account=predict_account,
            contracts=contracts,
            web3=web3,
        )

    @property
    def contracts(self) -> Contracts | None:
        """Access to contract instances (read-only)."""
        return self._contracts

    # --- Order Amount Calculation Methods ---

    def get_limit_order_amounts(self, data: LimitHelperInput) -> OrderAmounts:
        """
        Calculate the amounts for a LIMIT strategy order.

        Args:
            data: The input data containing side, price, and quantity.

        Returns:
            OrderAmounts with price_per_share, maker_amount, and taker_amount.

        Raises:
            InvalidQuantityError: If quantity_wei is less than 1e16 or price is invalid.
        """
        if data.price_per_share_wei <= 0:
            raise InvalidQuantityError("Invalid pricePerShareWei. Must be greater than 0.")
        if data.quantity_wei < int(1e16):
            raise InvalidQuantityError()

        # Truncate to significant digits for precision
        price = retain_significant_digits(data.price_per_share_wei, 3)
        qty = retain_significant_digits(data.quantity_wei, 5)

        if price != data.price_per_share_wei:
            self._logger.debug(
                "getLimitOrderAmounts truncated pricePerShareWei to 3 significant digits"
            )
        if qty != data.quantity_wei:
            self._logger.debug("getLimitOrderAmounts truncated quantityWei to 5 significant digits")

        if data.side == Side.BUY:
            return OrderAmounts(
                last_price=price,
                price_per_share=price,
                maker_amount=(price * qty) // self._precision,
                taker_amount=qty,
                amount=qty,
                slippage_bps=0,
                is_min_amount_out=False,
            )
        else:  # SELL
            return OrderAmounts(
                last_price=price,
                price_per_share=price,
                maker_amount=qty,
                taker_amount=(price * qty) // self._precision,
                amount=qty,
                slippage_bps=0,
                is_min_amount_out=False,
            )

    def _process_book(self, depths: list[DepthLevel], quantity_wei: int) -> ProcessedBookAmounts:
        """
        Process the order book to derive average price and last price for MARKET orders.

        Args:
            depths: Array of price levels and their quantities, sorted by price.
            quantity_wei: The total quantity of shares in wei.

        Returns:
            ProcessedBookAmounts containing quantity, price, and last_price.
        """
        result = ProcessedBookAmounts(quantity_wei=0, price_wei=0, last_price_wei=0)

        for price, qty in depths:
            remaining_qty_wei = quantity_wei - result.quantity_wei
            price_wei = float_to_wei(price, self._precision)
            qty_wei = float_to_wei(qty, self._precision)

            if remaining_qty_wei <= 0:
                break

            if remaining_qty_wei < qty_wei:
                result.quantity_wei += remaining_qty_wei
                # Accumulate price * qty without intermediate division to preserve precision
                result.price_wei += price_wei * remaining_qty_wei
                result.last_price_wei = price_wei
            else:
                result.quantity_wei += qty_wei
                # Accumulate price * qty without intermediate division to preserve precision
                result.price_wei += price_wei * qty_wei
                result.last_price_wei = price_wei

        return result

    def _get_market_order_amounts_by_quantity(
        self,
        data: MarketHelperInput,
        book: Book,
    ) -> OrderAmounts:
        """Calculate market order amounts by quantity."""
        qty = retain_significant_digits(data.quantity_wei, 5)

        if qty != data.quantity_wei:
            self._logger.debug(
                "getMarketOrderAmountsByQuantity truncated quantityWei to 5 significant digits"
            )

        if qty < int(1e16):
            raise InvalidQuantityError()

        slippage_bps = data.slippage_bps

        if data.side == Side.BUY:
            processed = self._process_book(book.asks, qty)
            is_min_amount_out = data.is_min_amount_out
            price_per_share = (
                processed.price_wei // processed.quantity_wei if processed.quantity_wei > 0 else 0
            )

            if is_min_amount_out:
                # makerAmount = expected cost (avg price * shares), not worstTierPrice * shares.
                # the signed ratio makerAmount/takerAmount equals worstTierPrice/(1-slippage),
                # which enables SPLIT (mint) matches at all book price levels while minimising
                # the USD commitment so users can spend their full wallet balance.
                maker_amount = processed.price_wei // self._precision
                # signedShares = expectedCost / worstTierPrice. fewer than actual shares,
                # but the OB fills up to `amount` (actual shares), constrained by the USD budget.
                signed_shares = (
                    processed.price_wei // processed.last_price_wei
                    if processed.last_price_wei > 0
                    else 0
                )
                taker_amount = (
                    max((signed_shares * (10_000 - slippage_bps)) // 10_000, 0)
                    if slippage_bps > 0
                    else signed_shares
                )
                return OrderAmounts(
                    last_price=processed.last_price_wei,
                    price_per_share=price_per_share,
                    maker_amount=maker_amount,
                    taker_amount=taker_amount,
                    amount=processed.quantity_wei,
                    slippage_bps=slippage_bps,
                    is_min_amount_out=is_min_amount_out,
                )

            # default: makerAmount = worstTierPrice * shares, inflated by slippage.
            # takerAmount = shares (unchanged).
            base_maker_amount = (
                processed.last_price_wei * processed.quantity_wei
            ) // self._precision
            maker_amount = (
                min(
                    (base_maker_amount * (10_000 + slippage_bps)) // 10_000,
                    processed.quantity_wei,
                )
                if slippage_bps > 0
                else base_maker_amount
            )
            return OrderAmounts(
                last_price=processed.last_price_wei,
                price_per_share=price_per_share,
                maker_amount=maker_amount,
                taker_amount=processed.quantity_wei,
                amount=processed.quantity_wei,
                slippage_bps=slippage_bps,
                is_min_amount_out=is_min_amount_out,
            )
        else:  # SELL
            processed = self._process_book(book.bids, qty)
            base_taker_amount = (
                processed.last_price_wei * processed.quantity_wei
            ) // self._precision
            # Floor at 0 to prevent underflow
            taker_amount = (
                max(
                    (base_taker_amount * (10_000 - slippage_bps)) // 10_000,
                    0,
                )
                if slippage_bps > 0
                else base_taker_amount
            )
            return OrderAmounts(
                last_price=processed.last_price_wei,
                # price_wei now contains sum of (price * qty) without division,
                # so divide by quantity only (no need to multiply by precision)
                price_per_share=processed.price_wei // processed.quantity_wei
                if processed.quantity_wei > 0
                else 0,
                maker_amount=processed.quantity_wei,
                taker_amount=taker_amount,
                amount=processed.quantity_wei,
                slippage_bps=slippage_bps,
                is_min_amount_out=False,
            )

    def _get_market_order_amounts_by_value(
        self,
        data: MarketHelperValueInput,
        book: Book,
    ) -> OrderAmounts:
        """Calculate market order amounts by value (BUY only)."""
        if data.value_wei < int(1e18):
            raise InvalidQuantityError()

        currency_amount_wei = data.value_wei
        number_of_shares = 0
        total_price = 0

        for price, qty in book.asks:
            price_wei = float_to_wei(price, self._precision)
            qty_wei = float_to_wei(qty, self._precision)

            remaining_spend = currency_amount_wei - total_price

            if remaining_spend <= 0:
                break

            tier_total_price = (price_wei * qty_wei) // self._precision

            # Check if the market buy can consume this entire price tier
            if tier_total_price <= remaining_spend:
                number_of_shares += qty_wei
                total_price += (price_wei * qty_wei) // self._precision
            else:
                # Consume as much as we can
                fractional_share_amount = (
                    (remaining_spend * self._precision) // price_wei if price_wei > 0 else 0
                )
                number_of_shares += fractional_share_amount
                total_price += (price_wei * fractional_share_amount) // self._precision

        rounded_shares = retain_significant_digits(number_of_shares, 5)
        amounts = self._get_market_order_amounts_by_quantity(
            MarketHelperInput(
                side=Side.BUY,
                quantity_wei=rounded_shares,
                slippage_bps=data.slippage_bps,
                is_min_amount_out=data.is_min_amount_out,
            ),
            book,
        )

        return OrderAmounts(
            price_per_share=amounts.price_per_share,
            maker_amount=amounts.maker_amount,
            taker_amount=amounts.taker_amount,
            amount=rounded_shares,
            last_price=amounts.last_price,
            slippage_bps=amounts.slippage_bps,
            is_min_amount_out=amounts.is_min_amount_out,
        )

    def get_market_order_amounts(
        self,
        data: MarketHelperInput | MarketHelperValueInput,
        book: Book,
    ) -> OrderAmounts:
        """
        Calculate the amounts for a MARKET strategy order.

        The order book should be retrieved from the `GET /orderbook/{marketId}` endpoint.

        Args:
            data: The input data (quantity or value based).
            book: The orderbook data.

        Returns:
            OrderAmounts with average price_per_share, maker_amount, and taker_amount.

        Raises:
            InvalidQuantityError: If quantity_wei is less than 1e16.
        """
        if isinstance(data, MarketHelperValueInput) and data.side == Side.BUY:
            return self._get_market_order_amounts_by_value(data, book)
        return self._get_market_order_amounts_by_quantity(data, book)

    # --- Order Building Methods ---

    def build_order(
        self,
        strategy: Literal["MARKET", "LIMIT"],
        data: BuildOrderInput,
    ) -> Order:
        """
        Build an order based on the provided strategy and data.

        The current `feeRateBps` should be fetched via the `GET /markets` endpoint.
        The expiration for market orders is ignored.

        Args:
            strategy: The order strategy ("MARKET" or "LIMIT").
            data: The order input data.

        Returns:
            A constructed Order object.

        Raises:
            InvalidExpirationError: If expiration is not in the future (LIMIT only).
            MakerSignerMismatchError: If maker and signer don't match.
        """
        # Default expiration: 2100-01-01 (arbitrary date for orders without expiration)
        expires_at = data.expires_at or datetime(2100, 1, 1, tzinfo=timezone.utc)
        limit_expiration = int(expires_at.timestamp())
        market_expiration = int(time.time()) + FIVE_MINUTES_SECONDS

        if self._predict_account and (
            data.maker != self._predict_account or data.signer != self._predict_account
        ):
            self._logger.warn("When using a Predict account the maker and signer are ignored.")

        if strategy == "MARKET" and data.expires_at:
            self._logger.warn("expiresAt for market orders is ignored.")

        if strategy != "MARKET" and expires_at.timestamp() <= time.time():
            raise InvalidExpirationError()

        signer_address = data.signer or (self._signer.address if self._signer else None)

        # Only validate maker/signer match when NOT using a predict_account
        # (when using predict_account, maker/signer from data are ignored)
        if not self._predict_account and data.maker and signer_address != data.maker:
            raise MakerSignerMismatchError()

        # Validate we have a valid signer/maker address
        effective_maker = self._predict_account or data.maker or signer_address
        effective_signer = self._predict_account or signer_address
        if not effective_maker or not effective_signer:
            raise MissingSignerError()

        return Order(
            salt=str(data.salt or self._generate_salt()),
            maker=effective_maker,
            signer=effective_signer,
            taker=data.taker or ZERO_ADDRESS,
            token_id=str(data.token_id),
            maker_amount=str(data.maker_amount),
            taker_amount=str(data.taker_amount),
            expiration=str(market_expiration if strategy == "MARKET" else limit_expiration),
            nonce=str(data.nonce or 0),
            fee_rate_bps=str(data.fee_rate_bps),
            side=data.side,
            signature_type=data.signature_type or SignatureType.EOA,
        )

    def _get_exchange_identifier(
        self,
        is_neg_risk: bool,
        is_yield_bearing: bool,
    ) -> str:
        """Get the exchange contract address based on market type."""
        if is_neg_risk:
            if is_yield_bearing:
                return self._addresses.YIELD_BEARING_NEG_RISK_CTF_EXCHANGE
            return self._addresses.NEG_RISK_CTF_EXCHANGE
        else:
            if is_yield_bearing:
                return self._addresses.YIELD_BEARING_CTF_EXCHANGE
            return self._addresses.CTF_EXCHANGE

    def build_typed_data(
        self,
        order: Order,
        *,
        is_neg_risk: bool,
        is_yield_bearing: bool,
    ) -> EIP712TypedData:
        """
        Build EIP-712 typed data for an order.

        The param `isNegRisk` can be found via the `GET /markets` or `GET /categories` endpoints.

        Args:
            order: The order to build typed data for.
            is_neg_risk: Whether this is a NegRisk market.
            is_yield_bearing: Whether this is a yield-bearing market.

        Returns:
            EIP712TypedData structure for signing.
        """
        verifying_contract = self._get_exchange_identifier(is_neg_risk, is_yield_bearing)

        return EIP712TypedData(
            primary_type="Order",
            types={
                "EIP712Domain": EIP712_DOMAIN,
                "Order": ORDER_STRUCTURE,
            },
            domain={
                "name": PROTOCOL_NAME,
                "version": PROTOCOL_VERSION,
                "chainId": self._chain_id,
                "verifyingContract": verifying_contract,
            },
            message=self._order_to_message(order),
        )

    def _order_to_message(self, order: Order) -> dict[str, Any]:
        """Convert Order to EIP-712 message format."""
        return {
            "salt": order.salt,
            "maker": order.maker,
            "signer": order.signer,
            "taker": order.taker,
            "tokenId": order.token_id,
            "makerAmount": order.maker_amount,
            "takerAmount": order.taker_amount,
            "expiration": order.expiration,
            "nonce": order.nonce,
            "feeRateBps": order.fee_rate_bps,
            "side": order.side,
            "signatureType": order.signature_type,
        }

    def _message_to_order(self, message: dict[str, Any]) -> Order:
        """Convert EIP-712 message to Order."""
        return Order(
            salt=str(message["salt"]),
            maker=message["maker"],
            signer=message["signer"],
            taker=message["taker"],
            token_id=str(message["tokenId"]),
            maker_amount=str(message["makerAmount"]),
            taker_amount=str(message["takerAmount"]),
            expiration=str(message["expiration"]),
            nonce=str(message["nonce"]),
            fee_rate_bps=str(message["feeRateBps"]),
            side=message["side"],
            signature_type=message["signatureType"],
        )

    def build_typed_data_hash(self, typed_data: EIP712TypedData) -> str:
        """
        Compute the hash of EIP-712 typed data.

        Args:
            typed_data: The typed data to hash.

        Returns:
            The hex-encoded hash string.

        Raises:
            FailedTypedDataEncoderError: If hashing fails.
        """
        try:
            structured_data = {
                "types": typed_data.types,
                "primaryType": typed_data.primary_type,
                "domain": typed_data.domain,
                "message": typed_data.message,
            }

            encoded = encode_typed_data(full_message=structured_data)
            return "0x" + _hash_eip191_message(encoded).hex()
        except Exception as e:
            raise FailedTypedDataEncoderError(e) from e

    def sign_typed_data_order(self, typed_data: EIP712TypedData) -> SignedOrder:
        """
        Sign an order using EIP-712 typed data.

        Args:
            typed_data: The typed data to sign.

        Returns:
            A SignedOrder with the signature attached.

        Raises:
            MissingSignerError: If no signer was provided.
            FailedOrderSignError: If signing fails.
        """
        if not self._signer:
            raise MissingSignerError()

        order = self._message_to_order(typed_data.message)

        try:
            if self._predict_account:
                hash_ = self.build_typed_data_hash(typed_data)
                signature = self.sign_predict_account_message({"raw": hash_})
            else:
                signature = self._sign_typed_data(typed_data)

            return SignedOrder(
                salt=order.salt,
                maker=order.maker,
                signer=order.signer,
                taker=order.taker,
                token_id=order.token_id,
                maker_amount=order.maker_amount,
                taker_amount=order.taker_amount,
                expiration=order.expiration,
                nonce=order.nonce,
                fee_rate_bps=order.fee_rate_bps,
                side=order.side,
                signature_type=order.signature_type,
                signature=signature,
            )
        except Exception as e:
            raise FailedOrderSignError(e) from e

    async def sign_typed_data_order_async(self, typed_data: EIP712TypedData) -> SignedOrder:
        """
        Sign an order using EIP-712 typed data (async).

        This is a convenience async wrapper around the sync method.
        Signing is CPU-bound, so this simply calls the sync version.

        Args:
            typed_data: The typed data to sign.

        Returns:
            A SignedOrder with the signature attached.

        Raises:
            MissingSignerError: If no signer was provided.
            FailedOrderSignError: If signing fails.
        """
        return self.sign_typed_data_order(typed_data)

    def _sign_typed_data(self, typed_data: EIP712TypedData) -> str:
        """Sign EIP-712 typed data with the signer."""
        if not self._signer:
            raise MissingSignerError()

        structured_data = {
            "types": typed_data.types,
            "primaryType": typed_data.primary_type,
            "domain": typed_data.domain,
            "message": typed_data.message,
        }

        encoded = encode_typed_data(full_message=structured_data)
        signed = self._signer.sign_message(encoded)
        return signed.signature.hex()

    def sign_predict_account_message(self, message: str | dict[str, str]) -> str:
        """
        Sign a message for a Predict account.

        Args:
            message: The message to sign (string or dict with 'raw' key for raw hash).

        Returns:
            The signature as a hex string.

        Raises:
            MissingSignerError: If no signer or predict_account was provided.
        """
        if not self._signer or not self._predict_account:
            raise MissingSignerError()

        validator_address = self._addresses.ECDSA_VALIDATOR
        kernel_domain = KERNEL_DOMAIN_BY_CHAIN_ID[self._chain_id]

        if isinstance(message, dict) and "raw" in message:
            message_hash = message["raw"]
        else:
            # Use EIP-191 prefix hash to match TS SDK's hashMessage()
            msg = encode_defunct(text=str(message))
            message_hash = "0x" + _hash_eip191_message(msg).hex()

        digest = eip712_wrap_hash(
            message_hash,
            {**kernel_domain, "verifyingContract": self._predict_account},
        )

        # Sign the digest
        message_bytes = (
            bytes.fromhex(digest[2:]) if digest.startswith("0x") else bytes.fromhex(digest)
        )
        signable_msg = encode_defunct(primitive=message_bytes)
        signed = self._signer.sign_message(signable_msg)

        # Concatenate: 0x01 + validator_address + signature
        return "0x01" + validator_address[2:] + signed.signature.hex()

    async def sign_predict_account_message_async(self, message: str | dict[str, str]) -> str:
        """
        Sign a message for a Predict account (async).

        This is a convenience async wrapper around the sync method.
        Signing is CPU-bound, so this simply calls the sync version.

        Args:
            message: The message to sign (string or dict with 'raw' key for raw hash).

        Returns:
            The signature as a hex string.

        Raises:
            MissingSignerError: If no signer or predict_account was provided.
        """
        return self.sign_predict_account_message(message)

    # --- Async Contract Interaction Methods ---

    def _encode_execution_calldata(self, to: str, calldata: str, value: int = 0) -> bytes:
        """
        Encode calldata for Kernel's execute function.

        Format: target_address (20 bytes) + value (32 bytes) + calldata
        Matches TS SDK implementation at OrderBuilder.ts:307-309

        Args:
            to: Target contract address
            calldata: Encoded function call (hex string with 0x prefix)
            value: ETH value to send (default 0)

        Returns:
            Encoded execution calldata as bytes
        """
        # Convert address to bytes (remove 0x, convert to bytes)
        address_bytes = bytes.fromhex(to[2:] if to.startswith("0x") else to)

        # Convert value to 32-byte big-endian representation
        value_bytes = value.to_bytes(32, byteorder="big")

        # Convert calldata to bytes
        calldata_bytes = bytes.fromhex(calldata[2:] if calldata.startswith("0x") else calldata)

        # Concatenate: address + value + calldata
        return address_bytes + value_bytes + calldata_bytes

    async def _handle_transaction_async(
        self,
        contract: Contract,
        method_name: str,
        *args: Any,
    ) -> TransactionResult:
        """Handle a transaction safely (async)."""
        if not self._contracts or not self._signer or not self._web3:
            raise MissingSignerError()

        try:
            method = getattr(contract.functions, method_name)(*args)

            # Estimate gas
            estimated_gas = method.estimate_gas({"from": self._signer.address})
            gas_limit = (estimated_gas * 125) // 100

            # Build transaction
            tx = method.build_transaction(
                {
                    "from": self._signer.address,
                    "nonce": self._web3.eth.get_transaction_count(self._signer.address, "pending"),
                    "gas": gas_limit,
                }
            )

            # Sign and send
            signed = self._signer.sign_transaction(tx)
            tx_hash = self._web3.eth.send_raw_transaction(signed.raw_transaction)
            receipt = self._web3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

            if receipt["status"] == 1:
                return TransactionSuccess(success=True, receipt=receipt)
            return TransactionFail(success=False, receipt=receipt)
        except Exception as e:
            return TransactionFail(success=False, cause=e)

    # --- Sync Wrappers ---

    def _run_async(self, coro: Coroutine[Any, Any, _T]) -> _T:
        """Run an async coroutine synchronously."""
        try:
            asyncio.get_running_loop()
            raise RuntimeError(
                "Cannot call sync method from within an async context. "
                "Use the async variant (e.g., method_async) instead."
            )
        except RuntimeError as e:
            if "no running event loop" not in str(e):
                raise
        return asyncio.run(coro)

    # --- Approval Methods ---

    def _approval_owner_address(self) -> str:
        """Return the account whose approvals and allowances are being managed."""
        if self._predict_account:
            return self._predict_account
        if self._signer:
            return self._signer.address
        raise MissingSignerError()

    async def set_ctf_exchange_approval_async(
        self,
        *,
        is_neg_risk: bool,
        is_yield_bearing: bool,
        approved: bool = True,
    ) -> TransactionResult:
        """
        Set ERC-1155 approval for the CTF Exchange (async).

        Args:
            is_neg_risk: Whether this is a NegRisk market.
            is_yield_bearing: Whether this is a yield-bearing market.
            approved: Whether to approve or revoke.

        Returns:
            TransactionResult indicating success or failure.
        """
        if not self._contracts:
            raise MissingSignerError()

        exchange_address = self._get_exchange_identifier(is_neg_risk, is_yield_bearing)
        ct_contract = get_conditional_tokens_contract(
            self._contracts,
            is_neg_risk=is_neg_risk,
            is_yield_bearing=is_yield_bearing,
        )

        # skip when the operator is already in the desired state (idempotent, saves gas).
        # a read failure is non-fatal: fall through so the send path reports any error.
        owner = self._approval_owner_address()
        try:
            is_approved = ct_contract.functions.isApprovedForAll(owner, exchange_address).call()
        except Exception:
            is_approved = None
        if is_approved == approved:
            return TransactionSuccess(success=True)

        if self._predict_account:
            encoded = ct_contract.encode_abi(
                abi_element_identifier="setApprovalForAll",
                args=[exchange_address, approved],
            )
            calldata = self._encode_execution_calldata(ct_contract.address, encoded, value=0)
            assert self._web3 is not None
            kernel_contract = make_contract(self._web3, self._predict_account, KERNEL_ABI)
            return await self._handle_transaction_async(
                kernel_contract, "execute", self._execution_mode, calldata
            )
        else:
            return await self._handle_transaction_async(
                ct_contract,
                "setApprovalForAll",
                exchange_address,
                approved,
            )

    def set_ctf_exchange_approval(
        self,
        *,
        is_neg_risk: bool,
        is_yield_bearing: bool,
        approved: bool = True,
    ) -> TransactionResult:
        """Set ERC-1155 approval for the CTF Exchange (sync)."""
        return self._run_async(
            self.set_ctf_exchange_approval_async(
                is_neg_risk=is_neg_risk,
                is_yield_bearing=is_yield_bearing,
                approved=approved,
            )
        )

    async def set_neg_risk_adapter_approval_async(
        self,
        *,
        is_yield_bearing: bool,
        approved: bool = True,
    ) -> TransactionResult:
        """Set ERC-1155 approval for the NegRisk Adapter (async)."""
        if not self._contracts:
            raise MissingSignerError()

        adapter_address = (
            self._addresses.YIELD_BEARING_NEG_RISK_ADAPTER
            if is_yield_bearing
            else self._addresses.NEG_RISK_ADAPTER
        )
        ct_contract = get_conditional_tokens_contract(
            self._contracts,
            is_neg_risk=True,
            is_yield_bearing=is_yield_bearing,
        )

        # skip when the operator is already in the desired state (idempotent, saves gas).
        # a read failure is non-fatal: fall through so the send path reports any error.
        owner = self._approval_owner_address()
        try:
            is_approved = ct_contract.functions.isApprovedForAll(owner, adapter_address).call()
        except Exception:
            is_approved = None
        if is_approved == approved:
            return TransactionSuccess(success=True)

        if self._predict_account:
            encoded = ct_contract.encode_abi(
                abi_element_identifier="setApprovalForAll",
                args=[adapter_address, approved],
            )
            calldata = self._encode_execution_calldata(ct_contract.address, encoded, value=0)
            assert self._web3 is not None
            kernel_contract = make_contract(self._web3, self._predict_account, KERNEL_ABI)
            return await self._handle_transaction_async(
                kernel_contract, "execute", self._execution_mode, calldata
            )
        else:
            return await self._handle_transaction_async(
                ct_contract,
                "setApprovalForAll",
                adapter_address,
                approved,
            )

    def set_neg_risk_adapter_approval(
        self,
        *,
        is_yield_bearing: bool,
        approved: bool = True,
    ) -> TransactionResult:
        """Set ERC-1155 approval for the NegRisk Adapter (sync)."""
        return self._run_async(
            self.set_neg_risk_adapter_approval_async(
                is_yield_bearing=is_yield_bearing,
                approved=approved,
            )
        )

    async def set_ctf_exchange_allowance_async(
        self,
        *,
        is_neg_risk: bool,
        is_yield_bearing: bool,
        amount: int = MAX_UINT256,
    ) -> TransactionResult:
        """Set ERC-20 (USDT) allowance for the CTF Exchange (async)."""
        if not self._contracts:
            raise MissingSignerError()

        exchange_address = self._get_exchange_identifier(is_neg_risk, is_yield_bearing)

        # skip when the existing allowance already covers the requested amount (idempotent).
        # a read failure is non-fatal: fall through so the send path reports any error.
        owner = self._approval_owner_address()
        try:
            current_allowance = self._contracts.usdt.functions.allowance(
                owner, exchange_address
            ).call()
        except Exception:
            current_allowance = None
        if current_allowance is not None and current_allowance >= amount:
            return TransactionSuccess(success=True)

        if self._predict_account:
            encoded = self._contracts.usdt.encode_abi(
                abi_element_identifier="approve",
                args=[exchange_address, amount],
            )
            calldata = self._encode_execution_calldata(
                self._contracts.usdt.address, encoded, value=0
            )
            assert self._web3 is not None
            kernel_contract = make_contract(self._web3, self._predict_account, KERNEL_ABI)
            return await self._handle_transaction_async(
                kernel_contract, "execute", self._execution_mode, calldata
            )
        else:
            return await self._handle_transaction_async(
                self._contracts.usdt,
                "approve",
                exchange_address,
                amount,
            )

    def set_ctf_exchange_allowance(
        self,
        *,
        is_neg_risk: bool,
        is_yield_bearing: bool,
        amount: int = MAX_UINT256,
    ) -> TransactionResult:
        """Set ERC-20 (USDT) allowance for the CTF Exchange (sync)."""
        return self._run_async(
            self.set_ctf_exchange_allowance_async(
                is_neg_risk=is_neg_risk,
                is_yield_bearing=is_yield_bearing,
                amount=amount,
            )
        )

    async def _set_track_approvals_async(self, is_yield_bearing: bool) -> list[TransactionResult]:
        """Set the five approvals required for a single market track (standard or yield-bearing)."""
        return [
            # Standard CTF Exchange
            await self.set_ctf_exchange_approval_async(
                is_neg_risk=False, is_yield_bearing=is_yield_bearing
            ),
            await self.set_ctf_exchange_allowance_async(
                is_neg_risk=False, is_yield_bearing=is_yield_bearing
            ),
            # NegRisk CTF Exchange
            await self.set_ctf_exchange_approval_async(
                is_neg_risk=True, is_yield_bearing=is_yield_bearing
            ),
            await self.set_ctf_exchange_allowance_async(
                is_neg_risk=True, is_yield_bearing=is_yield_bearing
            ),
            # NegRisk Adapter
            await self.set_neg_risk_adapter_approval_async(is_yield_bearing=is_yield_bearing),
        ]

    async def set_approvals_async(
        self,
        *,
        is_yield_bearing: bool | None = None,
    ) -> SetApprovalsResult:
        """
        Set all necessary approvals for trading (async).

        By default both the standard and yield-bearing contracts are approved, so the
        account is ready to trade every market type from a single call. The operations
        are idempotent: any approval already in place is detected on-chain and skipped,
        so calling this more than once only sends the transactions that are missing.

        Args:
            is_yield_bearing: ``None`` (default) approves both tracks. Pass ``True`` or
                ``False`` to limit the run to the yield-bearing or standard track only.

        Returns:
            SetApprovalsResult with overall success status and the per-operation results.
        """
        tracks: list[bool] = [False, True] if is_yield_bearing is None else [is_yield_bearing]

        results: list[TransactionResult] = []
        for track in tracks:
            results.extend(await self._set_track_approvals_async(track))

        success = all(r.success for r in results)
        return SetApprovalsResult(success=success, transactions=results)

    def set_approvals(
        self,
        *,
        is_yield_bearing: bool | None = None,
    ) -> SetApprovalsResult:
        """Set all necessary approvals for trading (sync). See ``set_approvals_async``."""
        return self._run_async(self.set_approvals_async(is_yield_bearing=is_yield_bearing))

    # --- Scoped Approvals ---

    def _exchange_key(self, is_neg_risk: bool, is_yield_bearing: bool) -> str:
        """Return the Addresses key of the exchange for a market type."""
        if is_neg_risk:
            return (
                "YIELD_BEARING_NEG_RISK_CTF_EXCHANGE"
                if is_yield_bearing
                else "NEG_RISK_CTF_EXCHANGE"
            )
        return "YIELD_BEARING_CTF_EXCHANGE" if is_yield_bearing else "CTF_EXCHANGE"

    def _ctf_key(self, is_neg_risk: bool, is_yield_bearing: bool) -> str:
        """Return the Addresses key of the conditional tokens contract for a market type."""
        if is_yield_bearing:
            return (
                "YIELD_BEARING_NEG_RISK_CONDITIONAL_TOKENS"
                if is_neg_risk
                else "YIELD_BEARING_CONDITIONAL_TOKENS"
            )
        return "NEG_RISK_CONDITIONAL_TOKENS" if is_neg_risk else "CONDITIONAL_TOKENS"

    def _adapter_key(self, is_yield_bearing: bool) -> str:
        """Return the Addresses key of the neg risk adapter for a track."""
        return "YIELD_BEARING_NEG_RISK_ADAPTER" if is_yield_bearing else "NEG_RISK_ADAPTER"

    def _make_approval_step(
        self,
        step_type: ApprovalStepType,
        spender_key: str,
        token_key: str,
    ) -> ApprovalStep:
        """Build a self-describing ApprovalStep from a spender/token key pair."""
        role = SPENDER_ROLE_BY_KEY.get(spender_key)
        copy = APPROVAL_STEP_COPY.get(f"{role}:{step_type}") if role else None
        return ApprovalStep(
            id=f"{step_type}:{spender_key}",
            type=step_type,
            spender=getattr(self._addresses, spender_key),
            token=getattr(self._addresses, token_key),
            label=copy["label"] if copy else "",
            description=copy["description"] if copy else "",
        )

    def get_approval_steps(self, scope: ApprovalScope) -> list[ApprovalStep]:
        """
        Return the minimal, ordered set of approvals required for an operation on a market type.

        Pure: requires no signer and performs no network calls. Operations that need no approval
        (e.g. a standard MERGE or REDEEM) return an empty list.

        Args:
            scope: The operation and market type to scope the approvals to.

        Returns:
            The ordered approval steps.

        Raises:
            InvalidApprovalOperationError: If CONVERT is requested for a non-neg-risk market.
        """
        exchange_key = self._exchange_key(scope.is_neg_risk, scope.is_yield_bearing)
        ctf_key = self._ctf_key(scope.is_neg_risk, scope.is_yield_bearing)
        adapter_key = self._adapter_key(scope.is_yield_bearing)

        def erc1155(spender_key: str) -> ApprovalStep:
            return self._make_approval_step("ERC1155_APPROVAL", spender_key, ctf_key)

        def erc20(spender_key: str) -> ApprovalStep:
            return self._make_approval_step("ERC20_ALLOWANCE", spender_key, "USDT")

        op = scope.operation
        if op == "TRADE":
            steps: list[ApprovalStep] = []
            include_sell = scope.side is None or scope.side == Side.SELL
            include_buy = scope.side is None or scope.side == Side.BUY
            if include_sell:
                steps.append(erc1155(exchange_key))
            # Neg risk matches route minting/merging through the adapter, which moves the
            # user's conditional tokens, so the adapter must be approved regardless of side.
            if scope.is_neg_risk:
                steps.append(erc1155(adapter_key))
            if include_buy:
                steps.append(erc20(exchange_key))
            return steps
        if op == "SPLIT":
            # splitPosition pulls USDT: from the adapter for neg risk, else from the CT contract.
            return [erc20(adapter_key)] if scope.is_neg_risk else [erc20(ctf_key)]
        if op == "MERGE":
            # Neg risk merges burn the user's tokens via the adapter; standard merges burn directly.
            return [erc1155(adapter_key)] if scope.is_neg_risk else []
        if op == "REDEEM":
            # Neg risk claims redeem via the adapter; standard redemptions burn the user's tokens.
            return [erc1155(adapter_key)] if scope.is_neg_risk else []
        if op == "CONVERT":
            if not scope.is_neg_risk:
                raise InvalidApprovalOperationError("CONVERT is only valid for neg-risk markets.")
            return [erc1155(adapter_key)]
        raise InvalidApprovalOperationError(f"Unknown approval operation: {op}")

    def get_all_approval_steps(
        self,
        *,
        is_yield_bearing: bool | None = None,
    ) -> list[ApprovalStep]:
        """
        Return every approval the protocol could require, deduplicated by id.

        Spans both market types (standard and neg risk) and, by default, both tracks (standard and
        yield-bearing). This is the per-step, progress-reportable equivalent of set_approvals()
        (and a slight superset, since it also includes the split allowances). Pure: needs no signer.

        Args:
            is_yield_bearing: Limit to a single track. When None (default), both tracks are included.

        Returns:
            The full, deduplicated list of approval steps.
        """
        tracks = [False, True] if is_yield_bearing is None else [is_yield_bearing]
        seen: set[str] = set()
        steps: list[ApprovalStep] = []

        for yb in tracks:
            for is_neg_risk in (False, True):
                # CONVERT is neg-risk only; its single step is already covered by the others.
                operations: list[ApprovalOperation] = (
                    ["TRADE", "SPLIT", "MERGE", "REDEEM", "CONVERT"]
                    if is_neg_risk
                    else ["TRADE", "SPLIT", "MERGE", "REDEEM"]
                )
                for operation in operations:
                    scope = ApprovalScope(
                        operation=operation, is_neg_risk=is_neg_risk, is_yield_bearing=yb
                    )
                    for step in self.get_approval_steps(scope):
                        if step.id not in seen:
                            seen.add(step.id)
                            steps.append(step)

        return steps

    def _resolve_token_contract(self, step: ApprovalStep) -> Contract:
        """Resolve the web3 contract an approval step acts on (USDT or a CT contract)."""
        if not self._contracts:
            raise MissingSignerError()
        if step.type == "ERC20_ALLOWANCE":
            return self._contracts.usdt
        target = Web3.to_checksum_address(step.token)
        for ct in (
            self._contracts.conditional_tokens,
            self._contracts.neg_risk_conditional_tokens,
            self._contracts.yield_bearing_conditional_tokens,
            self._contracts.yield_bearing_neg_risk_conditional_tokens,
        ):
            if ct.address == target:
                return ct
        raise ValueError(f"Unknown approval token: {step.token}")

    async def check_approval_async(self, step: ApprovalStep) -> bool:
        """Check whether a single approval step is already satisfied on-chain (async)."""
        if not self._contracts:
            raise MissingSignerError()

        owner = self._approval_owner_address()

        if step.type == "ERC20_ALLOWANCE":
            allowance: int = self._contracts.usdt.functions.allowance(owner, step.spender).call()
            return allowance >= MAX_INT256

        ct = self._resolve_token_contract(step)
        approved: bool = ct.functions.isApprovedForAll(owner, step.spender).call()
        return approved

    def check_approval(self, step: ApprovalStep) -> bool:
        """Check whether a single approval step is already satisfied on-chain (sync)."""
        return self._run_async(self.check_approval_async(step))

    async def check_approvals_async(self, steps: list[ApprovalStep]) -> list[ApprovalCheck]:
        """
        Check whether each approval step is already satisfied on-chain (async).

        Note: unlike the TypeScript SDK, this issues the reads sequentially (no multicall).
        """
        results: list[ApprovalCheck] = []
        for step in steps:
            satisfied = await self.check_approval_async(step)
            results.append(ApprovalCheck(step=step, satisfied=satisfied))
        return results

    def check_approvals(self, steps: list[ApprovalStep]) -> list[ApprovalCheck]:
        """Check whether each approval step is already satisfied on-chain (sync)."""
        return self._run_async(self.check_approvals_async(steps))

    async def _set_erc1155_approval_async(
        self, token_contract: Contract, spender: str, approved: bool
    ) -> TransactionResult:
        """Set ERC-1155 setApprovalForAll for an operator (EOA or Predict-account Kernel path)."""
        if self._predict_account:
            encoded = token_contract.encode_abi(
                abi_element_identifier="setApprovalForAll",
                args=[spender, approved],
            )
            calldata = self._encode_execution_calldata(token_contract.address, encoded, value=0)
            assert self._web3 is not None
            kernel_contract = make_contract(self._web3, self._predict_account, KERNEL_ABI)
            return await self._handle_transaction_async(
                kernel_contract, "execute", self._execution_mode, calldata
            )
        return await self._handle_transaction_async(
            token_contract, "setApprovalForAll", spender, approved
        )

    async def _set_erc20_allowance_async(self, spender: str, amount: int) -> TransactionResult:
        """Set ERC-20 (USDT) allowance for a spender (EOA or Predict-account Kernel path)."""
        assert self._contracts is not None
        usdt = self._contracts.usdt
        if self._predict_account:
            encoded = usdt.encode_abi(
                abi_element_identifier="approve",
                args=[spender, amount],
            )
            calldata = self._encode_execution_calldata(usdt.address, encoded, value=0)
            assert self._web3 is not None
            kernel_contract = make_contract(self._web3, self._predict_account, KERNEL_ABI)
            return await self._handle_transaction_async(
                kernel_contract, "execute", self._execution_mode, calldata
            )
        return await self._handle_transaction_async(usdt, "approve", spender, amount)

    async def set_approval_async(
        self,
        step: ApprovalStep,
        *,
        approved: bool = True,
        amount: int = MAX_UINT256,
    ) -> TransactionResult:
        """
        Execute a single approval step on-chain (async). Raw send: does not pre-check.

        Args:
            step: The step to execute.
            approved: ERC-1155: approve (default) or revoke (False). ERC-20: when False, revokes by
                setting the allowance to 0 (ignoring ``amount``).
            amount: ERC-20 only: the allowance to set when approving. Defaults to MAX_UINT256.

        Returns:
            The transaction result.
        """
        if not self._contracts:
            raise MissingSignerError()

        if step.type == "ERC1155_APPROVAL":
            token_contract = self._resolve_token_contract(step)
            return await self._set_erc1155_approval_async(token_contract, step.spender, approved)

        # For ERC-20, ``approved=False`` revokes by setting the allowance to 0.
        return await self._set_erc20_allowance_async(step.spender, amount if approved else 0)

    def set_approval(
        self,
        step: ApprovalStep,
        *,
        approved: bool = True,
        amount: int = MAX_UINT256,
    ) -> TransactionResult:
        """Execute a single approval step on-chain (sync)."""
        return self._run_async(self.set_approval_async(step, approved=approved, amount=amount))

    async def run_approvals_async(
        self,
        steps: list[ApprovalStep],
        *,
        skip_satisfied: bool = True,
        stop_on_error: bool = True,
        on_progress: Callable[[ApprovalProgress], None] | None = None,
    ) -> ApprovalRunReport:
        """
        Run the given approval steps in order, reporting progress (async).

        Duplicate steps (by id) are removed, so you can pass a union of scopes or a curated subset.
        Produce the steps with get_approval_steps(scope) (one operation) or get_all_approval_steps()
        (everything). By default each step is checked first and skipped if already satisfied, and the
        run stops on the first failure. Use check_approval + set_approval directly when you need finer
        control (e.g. gating each step on a user confirmation).

        Args:
            steps: The steps to run (e.g. from get_approval_steps / get_all_approval_steps).
            skip_satisfied: When True (default), skip steps already in place.
            stop_on_error: When True (default), stop after the first failure.
            on_progress: Optional callback invoked as each step transitions.

        Returns:
            The per-step report and overall success.
        """
        # Dedupe by id (first occurrence wins) so unioned/curated step lists "just work".
        seen: set[str] = set()
        unique_steps: list[ApprovalStep] = []
        for step in steps:
            if step.id not in seen:
                seen.add(step.id)
                unique_steps.append(step)

        results: list[ApprovalStepResult] = []
        success = True

        for step in unique_steps:
            if skip_satisfied:
                if on_progress:
                    on_progress(ApprovalProgress(step=step, status="checking"))
                # A pre-check read failure is non-fatal: fall through to the send path (matching
                # the legacy approval helpers) rather than aborting the whole run.
                try:
                    already_satisfied = await self.check_approval_async(step)
                except Exception:
                    already_satisfied = False
                if already_satisfied:
                    if on_progress:
                        on_progress(ApprovalProgress(step=step, status="skipped"))
                    results.append(ApprovalStepResult(step=step, status="skipped"))
                    continue

            if on_progress:
                on_progress(ApprovalProgress(step=step, status="submitting"))

            transaction = await self.set_approval_async(step)
            status: Literal["confirmed", "failed"] = (
                "confirmed" if transaction.success else "failed"
            )

            if on_progress:
                on_progress(ApprovalProgress(step=step, status=status, transaction=transaction))
            results.append(ApprovalStepResult(step=step, status=status, transaction=transaction))

            if not transaction.success:
                success = False
                if stop_on_error:
                    break

        return ApprovalRunReport(success=success, steps=results)

    def run_approvals(
        self,
        steps: list[ApprovalStep],
        *,
        skip_satisfied: bool = True,
        stop_on_error: bool = True,
        on_progress: Callable[[ApprovalProgress], None] | None = None,
    ) -> ApprovalRunReport:
        """Run the given approval steps in order (sync). See run_approvals_async."""
        return self._run_async(
            self.run_approvals_async(
                steps,
                skip_satisfied=skip_satisfied,
                stop_on_error=stop_on_error,
                on_progress=on_progress,
            )
        )

    # --- Balance Methods ---

    async def balance_of_async(
        self,
        token: Literal["USDT"] = "USDT",
        address: str | None = None,
    ) -> int:
        """
        Get the token balance for an address (async).

        Args:
            token: The token to check (currently only "USDT").
            address: The address to check (defaults to signer).

        Returns:
            The balance in wei.
        """
        if not self._contracts:
            raise MissingSignerError()

        check_address = (
            address or self._predict_account or (self._signer.address if self._signer else None)
        )
        if not check_address:
            raise MissingSignerError()

        result: int = self._contracts.usdt.functions.balanceOf(check_address).call()
        return result

    def balance_of(
        self,
        token: Literal["USDT"] = "USDT",
        address: str | None = None,
    ) -> int:
        """Get the token balance for an address (sync)."""
        return self._run_async(self.balance_of_async(token, address))

    # --- Redemption Methods ---

    async def redeem_positions_async(
        self,
        condition_id: str,
        index_set: Literal[1, 2],
        amount: int | None = None,
        *,
        is_neg_risk: bool,
        is_yield_bearing: bool,
    ) -> TransactionResult:
        """
        Redeem positions for a market (async).

        Args:
            condition_id: The condition ID.
            index_set: The index set (1 or 2).
            amount: The amount to redeem. Required for NegRisk markets.
            is_neg_risk: Whether this is a NegRisk (winner-takes-all) market.
            is_yield_bearing: Whether this is a yield-bearing market.

        Returns:
            TransactionResult indicating success or failure.

        Raises:
            MissingSignerError: If signer was not provided.
            ValueError: If amount is not provided for NegRisk markets.
        """
        if not self._contracts:
            raise MissingSignerError()

        if is_neg_risk:
            if amount is None:
                raise ValueError("amount is required for NegRisk markets")

            adapter_contract = get_neg_risk_adapter_contract(
                self._contracts,
                is_yield_bearing=is_yield_bearing,
            )
            amounts = [amount, 0] if index_set == 1 else [0, amount]

            if self._predict_account:
                encoded = adapter_contract.encode_abi(
                    abi_element_identifier="redeemPositions",
                    args=[condition_id, amounts],
                )
                calldata = self._encode_execution_calldata(
                    adapter_contract.address, encoded, value=0
                )
                assert self._web3 is not None
                kernel_contract = make_contract(self._web3, self._predict_account, KERNEL_ABI)
                return await self._handle_transaction_async(
                    kernel_contract, "execute", self._execution_mode, calldata
                )
            else:
                return await self._handle_transaction_async(
                    adapter_contract,
                    "redeemPositions",
                    condition_id,
                    amounts,
                )
        else:
            ct_contract = get_conditional_tokens_contract(
                self._contracts,
                is_neg_risk=False,
                is_yield_bearing=is_yield_bearing,
            )
            amounts = [index_set]

            if self._predict_account:
                encoded = ct_contract.encode_abi(
                    abi_element_identifier="redeemPositions",
                    args=[
                        self._addresses.USDT,
                        bytes.fromhex(ZERO_HASH[2:]),
                        condition_id,
                        amounts,
                    ],
                )
                calldata = self._encode_execution_calldata(ct_contract.address, encoded, value=0)
                assert self._web3 is not None
                kernel_contract = make_contract(self._web3, self._predict_account, KERNEL_ABI)
                return await self._handle_transaction_async(
                    kernel_contract, "execute", self._execution_mode, calldata
                )
            else:
                return await self._handle_transaction_async(
                    ct_contract,
                    "redeemPositions",
                    self._addresses.USDT,
                    bytes.fromhex(ZERO_HASH[2:]),
                    condition_id,
                    amounts,
                )

    def redeem_positions(
        self,
        condition_id: str,
        index_set: Literal[1, 2],
        amount: int | None = None,
        *,
        is_neg_risk: bool,
        is_yield_bearing: bool,
    ) -> TransactionResult:
        """Redeem positions for a market (sync)."""
        return self._run_async(
            self.redeem_positions_async(
                condition_id,
                index_set,
                amount,
                is_neg_risk=is_neg_risk,
                is_yield_bearing=is_yield_bearing,
            )
        )

    # --- Merge Positions Methods ---

    async def merge_positions_async(
        self,
        condition_id: str,
        amount: int,
        *,
        is_neg_risk: bool,
        is_yield_bearing: bool,
    ) -> TransactionResult:
        """
        Merge both outcome tokens back into collateral (USDT) (async).

        This combines both outcome tokens (YES and NO) back into the collateral token.
        Both outcome positions must have equal amounts to merge.

        Args:
            condition_id: The condition ID to merge positions for.
            amount: The amount of each outcome token to merge.
            is_neg_risk: Whether this is a NegRisk (winner-takes-all) market.
            is_yield_bearing: Whether this is a yield-bearing market.

        Returns:
            TransactionResult indicating success or failure.
        """
        if not self._contracts:
            raise MissingSignerError()

        if is_neg_risk:
            # NegRisk markets use the adapter contract
            adapter_contract = get_neg_risk_adapter_contract(
                self._contracts,
                is_yield_bearing=is_yield_bearing,
            )

            if self._predict_account:
                encoded = adapter_contract.encode_abi(
                    abi_element_identifier="mergePositions",
                    args=[condition_id, amount],
                )
                calldata = self._encode_execution_calldata(
                    adapter_contract.address, encoded, value=0
                )
                assert self._web3 is not None
                kernel_contract = make_contract(self._web3, self._predict_account, KERNEL_ABI)
                return await self._handle_transaction_async(
                    kernel_contract, "execute", self._execution_mode, calldata
                )
            else:
                return await self._handle_transaction_async(
                    adapter_contract,
                    "mergePositions",
                    condition_id,
                    amount,
                )
        else:
            # Standard markets use the conditional tokens contract
            ct_contract = get_conditional_tokens_contract(
                self._contracts,
                is_neg_risk=False,
                is_yield_bearing=is_yield_bearing,
            )
            partition = [1, 2]  # Both outcomes

            if self._predict_account:
                encoded = ct_contract.encode_abi(
                    abi_element_identifier="mergePositions",
                    args=[
                        self._addresses.USDT,
                        bytes.fromhex(ZERO_HASH[2:]),
                        condition_id,
                        partition,
                        amount,
                    ],
                )
                calldata = self._encode_execution_calldata(ct_contract.address, encoded, value=0)
                assert self._web3 is not None
                kernel_contract = make_contract(self._web3, self._predict_account, KERNEL_ABI)
                return await self._handle_transaction_async(
                    kernel_contract, "execute", self._execution_mode, calldata
                )
            else:
                return await self._handle_transaction_async(
                    ct_contract,
                    "mergePositions",
                    self._addresses.USDT,
                    bytes.fromhex(ZERO_HASH[2:]),
                    condition_id,
                    partition,
                    amount,
                )

    def merge_positions(
        self,
        condition_id: str,
        amount: int,
        *,
        is_neg_risk: bool,
        is_yield_bearing: bool,
    ) -> TransactionResult:
        """Merge both outcome tokens back into collateral (USDT) (sync)."""
        return self._run_async(
            self.merge_positions_async(
                condition_id, amount, is_neg_risk=is_neg_risk, is_yield_bearing=is_yield_bearing
            )
        )

    # --- Split Positions Methods ---

    async def split_positions_async(
        self,
        condition_id: str,
        amount: int,
        *,
        is_neg_risk: bool,
        is_yield_bearing: bool,
    ) -> TransactionResult:
        """
        Split collateral (USDT) into outcome tokens (async).

        This splits the collateral token into both outcome tokens for a condition.
        The amount specified will be converted into equal amounts of each outcome token.

        Args:
            condition_id: The condition ID to split positions for.
            amount: The amount of collateral to split into outcome tokens.
            is_neg_risk: Whether this is a NegRisk (winner-takes-all) market.
            is_yield_bearing: Whether this is a yield-bearing market.

        Returns:
            TransactionResult indicating success or failure.
        """
        if not self._contracts:
            raise MissingSignerError()

        if is_neg_risk:
            # NegRisk markets use the adapter contract
            adapter_contract = get_neg_risk_adapter_contract(
                self._contracts,
                is_yield_bearing=is_yield_bearing,
            )

            if self._predict_account:
                encoded = adapter_contract.encode_abi(
                    abi_element_identifier="splitPosition",
                    args=[condition_id, amount],
                )
                calldata = self._encode_execution_calldata(
                    adapter_contract.address, encoded, value=0
                )
                assert self._web3 is not None
                kernel_contract = make_contract(self._web3, self._predict_account, KERNEL_ABI)
                return await self._handle_transaction_async(
                    kernel_contract, "execute", self._execution_mode, calldata
                )
            else:
                return await self._handle_transaction_async(
                    adapter_contract,
                    "splitPosition",
                    condition_id,
                    amount,
                )
        else:
            # Standard markets use the conditional tokens contract
            ct_contract = get_conditional_tokens_contract(
                self._contracts,
                is_neg_risk=False,
                is_yield_bearing=is_yield_bearing,
            )
            partition = [1, 2]  # Both outcomes

            if self._predict_account:
                encoded = ct_contract.encode_abi(
                    abi_element_identifier="splitPosition",
                    args=[
                        self._addresses.USDT,
                        bytes.fromhex(ZERO_HASH[2:]),
                        condition_id,
                        partition,
                        amount,
                    ],
                )
                calldata = self._encode_execution_calldata(ct_contract.address, encoded, value=0)
                assert self._web3 is not None
                kernel_contract = make_contract(self._web3, self._predict_account, KERNEL_ABI)
                return await self._handle_transaction_async(
                    kernel_contract, "execute", self._execution_mode, calldata
                )
            else:
                return await self._handle_transaction_async(
                    ct_contract,
                    "splitPosition",
                    self._addresses.USDT,
                    bytes.fromhex(ZERO_HASH[2:]),
                    condition_id,
                    partition,
                    amount,
                )

    def split_positions(
        self,
        condition_id: str,
        amount: int,
        *,
        is_neg_risk: bool,
        is_yield_bearing: bool,
    ) -> TransactionResult:
        """Split collateral (USDT) into outcome tokens (sync)."""
        return self._run_async(
            self.split_positions_async(
                condition_id, amount, is_neg_risk=is_neg_risk, is_yield_bearing=is_yield_bearing
            )
        )

    # --- Cancel Orders Methods ---

    async def cancel_orders_async(
        self,
        orders: list[Order],
        options: CancelOrdersOptions,
    ) -> TransactionResult:
        """
        Cancel orders on the CTF Exchange (async).

        Args:
            orders: List of orders to cancel.
            options: Cancellation options.

        Returns:
            TransactionResult indicating success or failure.
        """
        if not self._contracts:
            raise MissingSignerError()

        exchange_contract = get_exchange_contract(
            self._contracts,
            is_neg_risk=options.is_neg_risk,
            is_yield_bearing=options.is_yield_bearing,
        )

        # Convert orders to the contract format
        order_structs = []
        for order in orders:
            order_structs.append(
                (
                    int(order.salt),
                    order.maker,
                    order.signer,
                    order.taker,
                    int(order.token_id),
                    int(order.maker_amount),
                    int(order.taker_amount),
                    int(order.expiration),
                    int(order.nonce),
                    int(order.fee_rate_bps),
                    order.side.value if hasattr(order.side, "value") else int(order.side),
                    order.signature_type.value
                    if hasattr(order.signature_type, "value")
                    else int(order.signature_type),
                    b"",  # Empty signature for cancellation
                )
            )

        if self._predict_account:
            encoded = exchange_contract.encode_abi(
                abi_element_identifier="cancelOrders", args=[order_structs]
            )

            calldata = self._encode_execution_calldata(
                exchange_contract.address,
                encoded,
                value=0,
            )

            assert self._web3 is not None
            kernel_contract = make_contract(self._web3, self._predict_account, KERNEL_ABI)

            return await self._handle_transaction_async(
                kernel_contract,
                "execute",
                self._execution_mode,
                calldata,
            )
        else:
            return await self._handle_transaction_async(
                exchange_contract,
                "cancelOrders",
                order_structs,
            )

    def cancel_orders(
        self,
        orders: list[Order],
        options: CancelOrdersOptions,
    ) -> TransactionResult:
        """Cancel orders on the CTF Exchange (sync)."""
        return self._run_async(self.cancel_orders_async(orders, options))

    # --- Token Validation ---

    async def validate_token_ids_async(
        self,
        token_ids: list[int],
        *,
        is_neg_risk: bool,
        is_yield_bearing: bool,
    ) -> bool:
        """
        Validate token IDs against the appropriate exchange (async).

        Args:
            token_ids: List of token IDs to validate.
            is_neg_risk: Whether to validate against NegRisk exchange.
            is_yield_bearing: Whether this is a yield-bearing market.

        Returns:
            True if all token IDs are valid.
        """
        if not self._contracts:
            raise MissingSignerError()

        exchange_contract = get_exchange_contract(
            self._contracts,
            is_neg_risk=is_neg_risk,
            is_yield_bearing=is_yield_bearing,
        )

        for token_id in token_ids:
            try:
                exchange_contract.functions.validateTokenId(token_id).call()
            except Exception:
                return False

        return True

    def validate_token_ids(
        self,
        token_ids: list[int],
        *,
        is_neg_risk: bool,
        is_yield_bearing: bool,
    ) -> bool:
        """Validate token IDs against the appropriate exchange (sync)."""
        return self._run_async(
            self.validate_token_ids_async(
                token_ids,
                is_neg_risk=is_neg_risk,
                is_yield_bearing=is_yield_bearing,
            )
        )
