def normalize_relative_path(value):
    if not isinstance(value, str):
        return None
    return value.replace("\\", "/")
