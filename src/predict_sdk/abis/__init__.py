"""Contract ABI exports for the Predict SDK."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

_ABI_DIR = Path(__file__).parent


def _load_abi(filename: str) -> list[dict[str, Any]]:
    """Load an ABI from a JSON file."""
    base_dir = _ABI_DIR.resolve()
    target_path = (base_dir / filename).resolve()
    try:
        target_path.relative_to(base_dir)
    except ValueError:
        raise Exception("Invalid file path")
    with open(target_path) as f:
        return cast(list[dict[str, Any]], json.load(f))


# Load all ABIs
CTF_EXCHANGE_ABI: list[dict[str, Any]] = _load_abi("CTFExchange.json")
NEG_RISK_CTF_EXCHANGE_ABI: list[dict[str, Any]] = _load_abi("NegRiskCtfExchange.json")
NEG_RISK_ADAPTER_ABI: list[dict[str, Any]] = _load_abi("NegRiskAdapter.json")
CONDITIONAL_TOKENS_ABI: list[dict[str, Any]] = _load_abi("ConditionalTokens.json")
YIELD_BEARING_CONDITIONAL_TOKENS_ABI: list[dict[str, Any]] = _load_abi(
    "YieldBearingConditionalTokens.json"
)
ERC20_ABI: list[dict[str, Any]] = _load_abi("ERC20.json")
KERNEL_ABI: list[dict[str, Any]] = _load_abi("Kernel.json")
ECDSA_VALIDATOR_ABI: list[dict[str, Any]] = _load_abi("ECDSAValidator.json")

__all__ = [
    "CTF_EXCHANGE_ABI",
    "NEG_RISK_CTF_EXCHANGE_ABI",
    "NEG_RISK_ADAPTER_ABI",
    "CONDITIONAL_TOKENS_ABI",
    "YIELD_BEARING_CONDITIONAL_TOKENS_ABI",
    "ERC20_ABI",
    "KERNEL_ABI",
    "ECDSA_VALIDATOR_ABI",
]
