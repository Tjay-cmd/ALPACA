"""Test connection to Alpaca paper trading API."""

import os
import sys

from dotenv import load_dotenv
from alpaca.trading.client import TradingClient

load_dotenv()

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")

PLACEHOLDER_VALUES = {"your_paper_key_id_here", "your_paper_secret_key_here"}


def main() -> None:
    if not API_KEY or not API_SECRET:
        print(
            "Error: Missing Alpaca API credentials.\n"
            "Set APCA_API_KEY_ID and APCA_API_SECRET_KEY in your .env file."
        )
        sys.exit(1)

    if API_KEY in PLACEHOLDER_VALUES or API_SECRET in PLACEHOLDER_VALUES:
        print(
            "Error: Alpaca API credentials are still placeholders.\n"
            "Replace them in .env with your real paper trading key and secret."
        )
        sys.exit(1)

    try:
        client = TradingClient(API_KEY, API_SECRET, paper=True)
        account = client.get_account()
    except Exception as exc:
        print(
            "Error: Could not connect to Alpaca. Check that your paper API key "
            f"and secret are valid.\nDetails: {exc}"
        )
        sys.exit(1)

    print("Connected to Alpaca paper trading.")
    print(f"Account status: {account.status}")
    print(f"Buying power:   {account.buying_power}")
    print(f"Cash balance:   {account.cash}")


if __name__ == "__main__":
    main()
