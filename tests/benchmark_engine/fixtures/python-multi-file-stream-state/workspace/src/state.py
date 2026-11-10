import json

from limits import clamp_sample


def parse_stream_state(payload, minimum, maximum):
    record = json.loads(payload)
    if not isinstance(record, dict):
        return None
    samples = record.get("samples", [])
    if not isinstance(samples, list):
        samples = []
    return {
        "state": record.get("state"),
        "samples": [
            clamp_sample(value, minimum, maximum)
            for value in samples
            if not isinstance(value, bool) and isinstance(value, (int, float))
        ],
    }
