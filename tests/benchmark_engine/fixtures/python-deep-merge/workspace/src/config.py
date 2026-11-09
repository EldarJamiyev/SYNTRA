def merge_config(base: dict, override: dict) -> dict:
    merged = dict(base)
    merged.update(override)
    return merged
