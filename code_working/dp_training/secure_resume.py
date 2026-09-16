"""Windows-DPAPI encrypted, atomic resume envelopes for the K5 research run."""

from __future__ import annotations

import ctypes
import hashlib
import io
import os
from ctypes import wintypes
from pathlib import Path
from typing import Any


MAGIC = b"NIH-CXR14-K5-RESUME-V1\0"
OPTIONAL_ENTROPY = b"nih-cxr14-k5-feasibility-resume-v1"
CRYPTPROTECT_UI_FORBIDDEN = 0x1


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


class DataBlob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]


def _blob(value: bytes) -> tuple[DataBlob, Any]:
    buffer = (ctypes.c_ubyte * len(value)).from_buffer_copy(value)
    return DataBlob(len(value), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte))), buffer


def _libraries() -> tuple[Any, Any]:
    require(os.name == "nt", "Windows DPAPI is required")
    crypt32 = ctypes.WinDLL("crypt32.dll", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32.dll", use_last_error=True)
    crypt32.CryptProtectData.argtypes = [
        ctypes.POINTER(DataBlob),
        wintypes.LPCWSTR,
        ctypes.POINTER(DataBlob),
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(DataBlob),
    ]
    crypt32.CryptProtectData.restype = wintypes.BOOL
    crypt32.CryptUnprotectData.argtypes = [
        ctypes.POINTER(DataBlob),
        ctypes.POINTER(wintypes.LPWSTR),
        ctypes.POINTER(DataBlob),
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(DataBlob),
    ]
    crypt32.CryptUnprotectData.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    return crypt32, kernel32


def protect_bytes(plaintext: bytes) -> bytes:
    require(bool(plaintext), "resume plaintext must be nonempty")
    crypt32, kernel32 = _libraries()
    plain_blob, plain_buffer = _blob(plaintext)
    entropy_blob, entropy_buffer = _blob(OPTIONAL_ENTROPY)
    output = DataBlob()
    ok = crypt32.CryptProtectData(
        ctypes.byref(plain_blob),
        "NIH CXR14 K5 research resume",
        ctypes.byref(entropy_blob),
        None,
        None,
        CRYPTPROTECT_UI_FORBIDDEN,
        ctypes.byref(output),
    )
    _ = (plain_buffer, entropy_buffer)
    if not ok:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return ctypes.string_at(output.pbData, output.cbData)
    finally:
        kernel32.LocalFree(output.pbData)


def unprotect_bytes(ciphertext: bytes) -> bytes:
    require(bool(ciphertext), "resume ciphertext must be nonempty")
    crypt32, kernel32 = _libraries()
    cipher_blob, cipher_buffer = _blob(ciphertext)
    entropy_blob, entropy_buffer = _blob(OPTIONAL_ENTROPY)
    output = DataBlob()
    description = wintypes.LPWSTR()
    ok = crypt32.CryptUnprotectData(
        ctypes.byref(cipher_blob),
        ctypes.byref(description),
        ctypes.byref(entropy_blob),
        None,
        None,
        CRYPTPROTECT_UI_FORBIDDEN,
        ctypes.byref(output),
    )
    _ = (cipher_buffer, entropy_buffer)
    if not ok:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return ctypes.string_at(output.pbData, output.cbData)
    finally:
        kernel32.LocalFree(output.pbData)
        if description:
            kernel32.LocalFree(description)


def serialize_payload(payload: dict[str, Any]) -> bytes:
    import torch

    require(isinstance(payload, dict), "resume payload must be a dictionary")
    buffer = io.BytesIO()
    torch.save(payload, buffer, pickle_protocol=2)
    body = buffer.getvalue()
    return MAGIC + hashlib.sha256(body).digest() + body


def deserialize_payload(plaintext: bytes) -> dict[str, Any]:
    import torch

    minimum = len(MAGIC) + hashlib.sha256().digest_size + 1
    require(len(plaintext) >= minimum, "resume plaintext is truncated")
    require(plaintext.startswith(MAGIC), "resume magic mismatch")
    offset = len(MAGIC)
    expected = plaintext[offset : offset + 32]
    body = plaintext[offset + 32 :]
    require(hashlib.sha256(body).digest() == expected, "resume plaintext hash mismatch")
    payload = torch.load(io.BytesIO(body), map_location="cpu", weights_only=True)
    require(isinstance(payload, dict), "decoded resume payload must be a dictionary")
    return payload


def load_envelope(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"resume envelope is missing: {path.name}")
    return deserialize_payload(unprotect_bytes(path.read_bytes()))


def save_envelope_atomic(current: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Encrypt in memory, fsync `.new`, verify, then rotate current/previous."""
    require(current.suffix == ".dpapi", "resume envelope must use .dpapi suffix")
    current.parent.mkdir(parents=True, exist_ok=True)
    temporary = current.with_name(current.name + ".new")
    previous = current.with_name(current.stem + ".previous" + current.suffix)
    require(not temporary.exists(), "stale .new resume envelope requires manual audit")
    encrypted = protect_bytes(serialize_payload(payload))
    try:
        with temporary.open("xb") as handle:
            handle.write(encrypted)
            handle.flush()
            os.fsync(handle.fileno())
        verified = load_envelope(temporary)
        require(
            verified.get("schema") == payload.get("schema")
            and verified.get("arm") == payload.get("arm")
            and verified.get("committed_step") == payload.get("committed_step"),
            "resume post-write verification mismatch",
        )
        if current.exists():
            os.replace(current, previous)
        os.replace(temporary, current)
    except BaseException:
        # An encrypted temporary is retained for audit if it was already written.
        raise
    require(current.is_file(), "atomic resume replacement failed")
    return {
        "ciphertext_sha256": sha256_file(current),
        "ciphertext_bytes": current.stat().st_size,
        "previous_retained": previous.is_file(),
        "plaintext_file_created": False,
    }
