def slugify(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip().lower().replace(" ", "-")
