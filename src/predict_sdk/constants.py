"""Constants for the Predict SDK."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Final

MAX_SALT: Final[int] = 2_147_483_648
FIVE_MINUTES_SECONDS: Final[int] = 60 * 5


class ChainId(IntEnum):
    """Supported chain IDs."""

    BNB_MAINNET = 56
    BNB_TESTNET = 97


class SignatureType(IntEnum):
    """
    Signature types for orders.

    Note: EOA also supports EIP-1271.
    """

    EOA = 0


class Side(IntEnum):
    """Order side."""

    BUY = 0
    SELL = 1


@dataclass(frozen=True)
class Addresses:
    """Contract addresses for a specific chain."""

    YIELD_BEARING_CTF_EXCHANGE: str
    YIELD_BEARING_NEG_RISK_CTF_EXCHANGE: str
    YIELD_BEARING_NEG_RISK_ADAPTER: str
    YIELD_BEARING_CONDITIONAL_TOKENS: str
    YIELD_BEARING_NEG_RISK_CONDITIONAL_TOKENS: str

    CTF_EXCHANGE: str
    NEG_RISK_CTF_EXCHANGE: str
    NEG_RISK_ADAPTER: str
    CONDITIONAL_TOKENS: str
    NEG_RISK_CONDITIONAL_TOKENS: str

    USDT: str
    KERNEL: str
    ECDSA_VALIDATOR: str


ADDRESSES_BY_CHAIN_ID: Final[dict[ChainId, Addresses]] = {
    ChainId.BNB_MAINNET: Addresses(
        YIELD_BEARING_CTF_EXCHANGE="0x6bEb5a40C032AFc305961162d8204CDA16DECFa5",
        YIELD_BEARING_NEG_RISK_CTF_EXCHANGE="0x8A289d458f5a134bA40015085A8F50Ffb681B41d",
        YIELD_BEARING_NEG_RISK_ADAPTER="0x41dCe1A4B8FB5e6327701750aF6231B7CD0B2A40",
        YIELD_BEARING_CONDITIONAL_TOKENS="0x9400F8Ad57e9e0F352345935d6D3175975eb1d9F",
        YIELD_BEARING_NEG_RISK_CONDITIONAL_TOKENS="0xF64b0b318AAf83BD9071110af24D24445719A07F",
        CTF_EXCHANGE="0x8BC070BEdAB741406F4B1Eb65A72bee27894B689",
        NEG_RISK_CTF_EXCHANGE="0x365fb81bd4A24D6303cd2F19c349dE6894D8d58A",
        NEG_RISK_ADAPTER="0xc3Cf7c252f65E0d8D88537dF96569AE94a7F1A6E",
        CONDITIONAL_TOKENS="0x22DA1810B194ca018378464a58f6Ac2B10C9d244",
        NEG_RISK_CONDITIONAL_TOKENS="0x22DA1810B194ca018378464a58f6Ac2B10C9d244",
        USDT="0x55d398326f99059fF775485246999027B3197955",
        KERNEL="0xBAC849bB641841b44E965fB01A4Bf5F074f84b4D",
        ECDSA_VALIDATOR="0x845ADb2C711129d4f3966735eD98a9F09fC4cE57",
    ),
    ChainId.BNB_TESTNET: Addresses(
        YIELD_BEARING_CTF_EXCHANGE="0x8a6B4Fa700A1e310b106E7a48bAFa29111f66e89",
        YIELD_BEARING_NEG_RISK_CTF_EXCHANGE="0x95D5113bc50eD201e319101bbca3e0E250662fCC",
        YIELD_BEARING_NEG_RISK_ADAPTER="0xb74aea04bdeBE912Aa425bC9173F9668e6f11F99",
        YIELD_BEARING_CONDITIONAL_TOKENS="0x38BF1cbD66d174bb5F3037d7068E708861D68D7f",
        YIELD_BEARING_NEG_RISK_CONDITIONAL_TOKENS="0x26e865CbaAe99b62fbF9D18B55c25B5E079A93D5",
        CTF_EXCHANGE="0x2A6413639BD3d73a20ed8C95F634Ce198ABbd2d7",
        NEG_RISK_CTF_EXCHANGE="0xd690b2bd441bE36431F6F6639D7Ad351e7B29680",
        NEG_RISK_ADAPTER="0x285c1B939380B130D7EBd09467b93faD4BA623Ed",
        CONDITIONAL_TOKENS="0x2827AAef52D71910E8FBad2FfeBC1B6C2DA37743",
        NEG_RISK_CONDITIONAL_TOKENS="0x2827AAef52D71910E8FBad2FfeBC1B6C2DA37743",
        USDT="0xB32171ecD878607FFc4F8FC0bCcE6852BB3149E0",
        KERNEL="0xBAC849bB641841b44E965fB01A4Bf5F074f84b4D",
        ECDSA_VALIDATOR="0x845ADb2C711129d4f3966735eD98a9F09fC4cE57",
    ),
}

RPC_URLS_BY_CHAIN_ID: Final[dict[ChainId, str]] = {
    ChainId.BNB_MAINNET: "https://bsc-dataseed.bnbchain.org/",
    ChainId.BNB_TESTNET: "https://bsc-testnet-dataseed.bnbchain.org/",
}

KERNEL_DOMAIN_BY_CHAIN_ID: Final[dict[ChainId, dict[str, str | int]]] = {
    ChainId.BNB_MAINNET: {
        "name": "Kernel",
        "version": "0.3.1",
        "chainId": ChainId.BNB_MAINNET,
    },
    ChainId.BNB_TESTNET: {
        "name": "Kernel",
        "version": "0.3.1",
        "chainId": ChainId.BNB_TESTNET,
    },
}

PROTOCOL_NAME: Final[str] = "predict.fun CTF Exchange"
PROTOCOL_VERSION: Final[str] = "1"

# EIP-712 Type Definitions
EIP712_DOMAIN: Final[list[dict[str, str]]] = [
    {"name": "name", "type": "string"},
    {"name": "version", "type": "string"},
    {"name": "chainId", "type": "uint256"},
    {"name": "verifyingContract", "type": "address"},
]

ORDER_STRUCTURE: Final[list[dict[str, str]]] = [
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
]

# Zero values
ZERO_ADDRESS: Final[str] = "0x" + "0" * 40
ZERO_HASH: Final[str] = "0x" + "0" * 64

# Max values
MAX_UINT256: Final[int] = 2**256 - 1
