"""
Encrypted storage for biometric arrays (templates, raw recordings, face
embedding), using the Windows Data Protection API (DPAPI).

DPAPI encrypts with a key derived from the logged-in Windows account, so the
files can only be decrypted by the same account on the same computer: a copied
templates/ folder is unreadable elsewhere. Tampering is detected (decryption
fails). No extra Python package is needed (ctypes + crypt32.dll).

File names do not change. An encrypted file starts with MAGIC; plain .npy
files (profiles saved before encryption existed, or on systems without
DPAPI) are still read. New files are written encrypted when DPAPI is
available; migrate_profiles.py --encrypt converts existing profiles.

Limits: anything running as the same Windows user can decrypt the files
(DPAPI protects against other accounts and other machines, not against
malware in the user's session). config.json (thresholds and summary
statistics) stays plain JSON.
"""

import io
import os
import sys

import numpy as np

MAGIC = b"WLDPAPI1"
ENTROPY = b"WaveLock biometric template v1"   # extra secret bound into each blob


class SecureStoreError(ValueError):   # ValueError: callers already treat it as "profile unreadable"
    """The file is encrypted but cannot be decrypted here (other account / PC, or damaged)."""


# ---------------------------------------------------------------- DPAPI (ctypes)
_dpapi = None


def _load_dpapi():
    global _dpapi
    if _dpapi is not None:
        return _dpapi or None
    if sys.platform != "win32":
        _dpapi = False
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class Blob(ctypes.Structure):
            _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

        crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        crypt32.CryptProtectData.argtypes = [ctypes.POINTER(Blob), wintypes.LPCWSTR, ctypes.POINTER(Blob),
                                             ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD,
                                             ctypes.POINTER(Blob)]
        crypt32.CryptProtectData.restype = wintypes.BOOL
        crypt32.CryptUnprotectData.argtypes = [ctypes.POINTER(Blob), ctypes.c_void_p, ctypes.POINTER(Blob),
                                               ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD,
                                               ctypes.POINTER(Blob)]
        crypt32.CryptUnprotectData.restype = wintypes.BOOL
        kernel32.LocalFree.argtypes = [ctypes.c_void_p]
        _dpapi = (ctypes, Blob, crypt32, kernel32)
    except (OSError, AttributeError):
        _dpapi = False
    return _dpapi or None


def available():
    """True when DPAPI encryption can be used on this system."""
    return _load_dpapi() is not None


def _call(protect, data):
    ctypes, Blob, crypt32, kernel32 = _load_dpapi()
    buf_in = ctypes.create_string_buffer(data, len(data))
    ent = ctypes.create_string_buffer(ENTROPY, len(ENTROPY))
    blob_in = Blob(len(data), ctypes.cast(buf_in, ctypes.POINTER(ctypes.c_char)))
    blob_ent = Blob(len(ENTROPY), ctypes.cast(ent, ctypes.POINTER(ctypes.c_char)))
    blob_out = Blob()
    CRYPTPROTECT_UI_FORBIDDEN = 0x1
    if protect:
        ok = crypt32.CryptProtectData(ctypes.byref(blob_in), "WaveLock", ctypes.byref(blob_ent),
                                      None, None, CRYPTPROTECT_UI_FORBIDDEN, ctypes.byref(blob_out))
    else:
        ok = crypt32.CryptUnprotectData(ctypes.byref(blob_in), None, ctypes.byref(blob_ent),
                                        None, None, CRYPTPROTECT_UI_FORBIDDEN, ctypes.byref(blob_out))
    if not ok:
        raise SecureStoreError(f"DPAPI {'encryption' if protect else 'decryption'} failed "
                               f"(Windows error {ctypes.get_last_error()}).")
    try:
        return ctypes.string_at(blob_out.pbData, blob_out.cbData)
    finally:
        kernel32.LocalFree(blob_out.pbData)


# ---------------------------------------------------------------- arrays
def encode_array(array, encrypt=None):
    """Bytes of a .npy file; encrypted (MAGIC + DPAPI blob) when possible."""
    bio = io.BytesIO()
    np.save(bio, np.asarray(array), allow_pickle=False)
    raw = bio.getvalue()
    if encrypt is None:
        encrypt = available()
    if not encrypt:
        return raw
    if not available():
        raise SecureStoreError("Encryption requested but DPAPI is not available on this system.")
    return MAGIC + _call(True, raw)


def decode_array(data):
    """Inverse of encode_array: accepts encrypted or plain .npy bytes."""
    if data.startswith(MAGIC):
        if not available():
            raise SecureStoreError("This profile is encrypted with Windows DPAPI and cannot be read here.")
        try:
            data = _call(False, data[len(MAGIC):])
        except SecureStoreError as e:
            raise SecureStoreError("Cannot decrypt the profile: it belongs to another Windows account "
                                   "or computer, or the file is damaged.") from e
    return np.load(io.BytesIO(data), allow_pickle=False)


def is_encrypted(path):
    with open(path, "rb") as f:
        return f.read(len(MAGIC)) == MAGIC


def save_array(path_or_file, array, encrypt=None):
    """np.save replacement (path or open binary file)."""
    data = encode_array(array, encrypt)
    if hasattr(path_or_file, "write"):
        path_or_file.write(data)
    else:
        with open(path_or_file, "wb") as f:
            f.write(data)


def load_array(path):
    """np.load replacement for profile files (encrypted or plain)."""
    with open(path, "rb") as f:
        return decode_array(f.read())


def encrypt_file_in_place(path):
    """Re-write a plain profile file encrypted (atomically). Returns True if changed."""
    if is_encrypted(path):
        return False
    array = load_array(path)
    tmp = path + ".tmp_enc"
    try:
        save_array(tmp, array, encrypt=True)
        if not np.array_equal(load_array(tmp), array):      # round-trip check before replacing
            raise SecureStoreError(f"Round-trip check failed for {path}")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return True
