"""One-time backfill of risk_check_passed / risk_notes on trade_log.csv."""

from execute import migrate_trade_log


if __name__ == "__main__":
    migrated = migrate_trade_log()
    if migrated == 0:
        print("Already migrated, or no trade_log.csv to update.")
