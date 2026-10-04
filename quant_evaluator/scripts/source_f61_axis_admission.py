"""Bound raw F61 axes before the registered forward-price alignment."""


def validate_f61_raw_axes(dates, assets, records, schema):
    """Allow only the bounded t/t+1/t+2 tail, not arbitrary larger inputs.

    ``load_labels`` owns calendar/price alignment. Callers must still enforce
    exact ``schema.shape`` after that alignment; source row receipts stay raw.
    """
    times, names, factors = schema.shape
    if not (times <= len(dates) <= times + 2
            and len(assets) == names and len(records) == factors):
        raise SystemExit("axis index does not describe the fixed F61 raw alignment domain")
