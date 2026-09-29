"""The wire codec of docs/env-protocol.md: frames in and out, and a batch's payload."""

import json
import struct
from typing import BinaryIO

import numpy as np

PROTOCOL = 1
FACT_FIELDS = (
    "scoreGained", "ate", "kills", "alive", "deathCause", "deathBy",
    "ended", "outcome", "reason", "ticks", "seat",
)

_HEAD = struct.Struct("<II")


class EnvError(Exception):
    """The engine refused a message (an `error` response)."""


class EnvProcessError(EnvError):
    """The subprocess died, closed its output or spoke nonsense."""


def encode_frame(header: dict, payload: bytes = b"") -> bytes:
    body = json.dumps(header).encode("utf-8")
    return _HEAD.pack(len(body), len(payload)) + body + payload


def _read_exact(stream: BinaryIO, count: int) -> bytes:
    data = stream.read(count)  # a buffered read returns fewer bytes only at EOF
    if data is None or len(data) < count:
        raise EOFError
    return data


def read_frame(stream: BinaryIO) -> tuple[dict, bytes]:
    """Blocks for one frame. Raises EOFError if the stream ends first."""
    json_length, payload_length = _HEAD.unpack(_read_exact(stream, _HEAD.size))
    header = json.loads(_read_exact(stream, json_length).decode("utf-8"))
    return header, _read_exact(stream, payload_length) if payload_length else b""


def decode_batch(header: dict, payload: bytes) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Returns (facts int32 [n, F], observations float32 [n, obsSize], the fact field names)."""
    if header.get("protocol") != PROTOCOL:
        raise EnvProcessError(f"unsupported protocol {header.get('protocol')!r}")
    n, obs_size = header["matches"], header["obsSize"]
    fields = tuple(header["factFields"])
    fact_bytes = n * len(fields) * 4
    if len(payload) != fact_bytes + n * obs_size * 4:
        raise EnvProcessError(f"batch payload is {len(payload)} bytes, expected {fact_bytes + n * obs_size * 4}")
    facts = np.frombuffer(payload, dtype="<i4", count=n * len(fields)).reshape(n, len(fields))
    obs = np.frombuffer(payload, dtype="<f4", offset=fact_bytes).reshape(n, obs_size)
    return facts.copy(), obs.astype(np.float32, copy=True), fields
