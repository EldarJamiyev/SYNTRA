def lookup_header(headers: dict, name: str, default=None):
    return headers.get(name, default)
