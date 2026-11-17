"""ctypes bindings to build/cmcp.dll for CMCP binary framing, TLV, LZ/RLE compression, CRC32C, and compat."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

# Constants matching core/include/cmcp/cmcp.h
CMCP_PROTOCOL_VERSION = 1
CMCP_HEADER_SIZE = 64

CMCP_FLAG_NONE = 0x0000
CMCP_FLAG_COMPRESSED = 0x0001
CMCP_FLAG_HAS_KV_DESC = 0x0002
CMCP_FLAG_STRICT_KV = 0x0004

CMCP_OK = 0
CMCP_ERR_INVALID_ARG = -1
CMCP_ERR_BUFFER_TOO_SMALL = -2
CMCP_ERR_BAD_MAGIC = -3
CMCP_ERR_UNSUPPORTED_VERSION = -4
CMCP_ERR_HEADER_CRC = -5
CMCP_ERR_PAYLOAD_CRC = -6
CMCP_ERR_CORRUPT_DATA = -7
CMCP_ERR_INCOMPATIBLE = -8
CMCP_ERR_IO = -9
CMCP_ERR_LIMIT_EXCEEDED = -10

CMCP_MSG_HELLO = 0x0001
CMCP_MSG_CONTEXT_TRANSFER = 0x0002
CMCP_MSG_KV_SNAPSHOT = 0x0003
CMCP_MSG_LATENT_TRANSFER = 0x0004
CMCP_MSG_ACK = 0x0005
CMCP_MSG_REJECT = 0x0006

CMCP_TIER_0_EXACT_KV = 0x0000
CMCP_TIER_1_SAME_FAMILY = 0x0001
CMCP_TIER_2_CROSS_MODEL = 0x0002
CMCP_TIER_3_TEXT_FALLBACK = 0x0003
CMCP_TIER_INCOMPATIBLE = 0xFFFF

CMCP_TLV_CONCEPT = 0x01
CMCP_TLV_ENTITY = 0x02
CMCP_TLV_RELATION = 0x03
CMCP_TLV_INSTRUCTION = 0x04
CMCP_TLV_CONSTRAINT = 0x05
CMCP_TLV_EVIDENCE = 0x06
CMCP_TLV_REFERENCE = 0x07
CMCP_TLV_DECISION = 0x08
CMCP_TLV_LATENT_TENSOR = 0x09
CMCP_TLV_KV_COMPAT_DESC = 0x0A


class CMCPError(RuntimeError):
    """Raised when cmcp.dll returns a non-zero error status."""

    def __init__(self, status: int, message: str = "") -> None:
        self.status = status
        super().__init__(message or f"CMCP C library error (status={status})")


class CMCPHeader(ctypes.Structure):
    _fields_ = [
        ("magic", ctypes.c_uint8 * 4),
        ("version", ctypes.c_uint16),
        ("flags", ctypes.c_uint16),
        ("msg_type", ctypes.c_uint16),
        ("compat_tier", ctypes.c_uint16),
        ("sender_model_hash", ctypes.c_uint64),
        ("receiver_model_hash", ctypes.c_uint64),
        ("tokenizer_hash", ctypes.c_uint64),
        ("payload_bytes", ctypes.c_uint32),
        ("raw_bytes", ctypes.c_uint32),
        ("sequence_id", ctypes.c_uint64),
        ("reserved", ctypes.c_uint32),
        ("payload_crc32c", ctypes.c_uint32),
        ("header_crc32c", ctypes.c_uint32),
    ]


class CMCPTLVRecord(ctypes.Structure):
    _fields_ = [
        ("tag", ctypes.c_uint8),
        ("flags", ctypes.c_uint8),
        ("length", ctypes.c_uint32),
        ("value", ctypes.POINTER(ctypes.c_uint8)),
    ]


class CMCPTLVWriter(ctypes.Structure):
    _fields_ = [
        ("buf", ctypes.POINTER(ctypes.c_uint8)),
        ("cap", ctypes.c_size_t),
        ("pos", ctypes.c_size_t),
    ]


class CMCPTLVReader(ctypes.Structure):
    _fields_ = [
        ("buf", ctypes.POINTER(ctypes.c_uint8)),
        ("len", ctypes.c_size_t),
        ("pos", ctypes.c_size_t),
    ]


class CMCPModelProfile(ctypes.Structure):
    _fields_ = [
        ("arch_name", ctypes.c_char * 64),
        ("arch_hash", ctypes.c_uint64),
        ("weights_hash", ctypes.c_uint64),
        ("tokenizer_hash", ctypes.c_uint64),
        ("n_layer", ctypes.c_uint32),
        ("n_embd", ctypes.c_uint32),
        ("n_head_kv", ctypes.c_uint32),
        ("rope_type", ctypes.c_int32),
        ("rope_freq_scale", ctypes.c_float),
        ("kv_dtype", ctypes.c_uint32),
        ("n_ctx", ctypes.c_uint32),
        ("pos_offset", ctypes.c_uint32),
    ]


class CMCPCompatResult(ctypes.Structure):
    _fields_ = [
        ("tier", ctypes.c_int),
        ("kv_compatible", ctypes.c_int),
        ("reason_code", ctypes.c_int),
        ("reason", ctypes.c_char * 128),
    ]


def _load_cmcp_dll() -> ctypes.CDLL:
    repo_root = Path(__file__).resolve().parents[2]
    dll_path = repo_root / "build" / "cmcp.dll"
    if not dll_path.exists():
        raise FileNotFoundError(f"cmcp.dll not found at {dll_path}")

    msys_bin = Path(r"C:\msys64\clang64\bin")
    if os.name == "nt" and msys_bin.exists():
        os.add_dll_directory(str(msys_bin))
    if os.name == "nt":
        os.add_dll_directory(str(dll_path.parent))

    lib = ctypes.CDLL(str(dll_path))

    lib.cmcp_crc32c.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_size_t]
    lib.cmcp_crc32c.restype = ctypes.c_uint32

    lib.cmcp_fnv1a_64.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.cmcp_fnv1a_64.restype = ctypes.c_uint64

    lib.cmcp_header_init.argtypes = [ctypes.POINTER(CMCPHeader)]
    lib.cmcp_header_init.restype = None

    lib.cmcp_header_encode.argtypes = [
        ctypes.POINTER(CMCPHeader),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint8),
    ]
    lib.cmcp_header_encode.restype = ctypes.c_int

    lib.cmcp_header_decode.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(CMCPHeader),
    ]
    lib.cmcp_header_decode.restype = ctypes.c_int

    lib.cmcp_verify_payload.argtypes = [
        ctypes.POINTER(CMCPHeader),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
    ]
    lib.cmcp_verify_payload.restype = ctypes.c_int

    lib.cmcp_tlv_writer_init.argtypes = [
        ctypes.POINTER(CMCPTLVWriter),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
    ]
    lib.cmcp_tlv_writer_init.restype = None

    lib.cmcp_tlv_write.argtypes = [
        ctypes.POINTER(CMCPTLVWriter),
        ctypes.c_uint8,
        ctypes.c_uint8,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    lib.cmcp_tlv_write.restype = ctypes.c_int

    lib.cmcp_tlv_reader_init.argtypes = [
        ctypes.POINTER(CMCPTLVReader),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
    ]
    lib.cmcp_tlv_reader_init.restype = None

    lib.cmcp_tlv_next.argtypes = [
        ctypes.POINTER(CMCPTLVReader),
        ctypes.POINTER(CMCPTLVRecord),
    ]
    lib.cmcp_tlv_next.restype = ctypes.c_int

    lib.cmcp_compress_bound.argtypes = [ctypes.c_size_t]
    lib.cmcp_compress_bound.restype = ctypes.c_size_t

    lib.cmcp_compress.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.cmcp_compress.restype = ctypes.c_int

    lib.cmcp_decompress.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.cmcp_decompress.restype = ctypes.c_int

    lib.cmcp_model_profile_init.argtypes = [
        ctypes.POINTER(CMCPModelProfile),
        ctypes.c_char_p,
    ]
    lib.cmcp_model_profile_init.restype = None

    lib.cmcp_check_kv_compat.argtypes = [
        ctypes.POINTER(CMCPModelProfile),
        ctypes.POINTER(CMCPModelProfile),
        ctypes.POINTER(CMCPCompatResult),
    ]
    lib.cmcp_check_kv_compat.restype = ctypes.c_int

    lib.cmcp_check_compat.argtypes = [
        ctypes.POINTER(CMCPModelProfile),
        ctypes.POINTER(CMCPModelProfile),
        ctypes.c_int,
        ctypes.POINTER(CMCPCompatResult),
    ]
    lib.cmcp_check_compat.restype = ctypes.c_int

    return lib


_LIB = _load_cmcp_dll()


def _as_uint8_ptr(data: bytes) -> Tuple[ctypes.Array, ctypes.POINTER(ctypes.c_uint8)]:
    arr = (ctypes.c_uint8 * max(1, len(data))).from_buffer_copy(data if data else b"\x00")
    return arr, ctypes.cast(arr, ctypes.POINTER(ctypes.c_uint8))


def cmcp_crc32c(data: bytes, init_crc: int = 0) -> int:
    if not data:
        return _LIB.cmcp_crc32c(init_crc, None, 0)
    buf = ctypes.create_string_buffer(data, len(data))
    return int(_LIB.cmcp_crc32c(init_crc, ctypes.cast(buf, ctypes.c_void_p), len(data)))


def cmcp_fnv1a_64(data: bytes) -> int:
    if not data:
        return int(_LIB.cmcp_fnv1a_64(None, 0))
    buf = ctypes.create_string_buffer(data, len(data))
    return int(_LIB.cmcp_fnv1a_64(ctypes.cast(buf, ctypes.c_void_p), len(data)))


def cmcp_header_encode(
    payload: bytes,
    msg_type: int = CMCP_MSG_CONTEXT_TRANSFER,
    compat_tier: int = CMCP_TIER_2_CROSS_MODEL,
    flags: int = CMCP_FLAG_NONE,
    sender_model_hash: int = 0,
    receiver_model_hash: int = 0,
    tokenizer_hash: int = 0,
    raw_bytes: Optional[int] = None,
    sequence_id: int = 1,
) -> Tuple[CMCPHeader, bytes]:
    hdr = CMCPHeader()
    _LIB.cmcp_header_init(ctypes.byref(hdr))
    hdr.flags = flags
    hdr.msg_type = msg_type
    hdr.compat_tier = compat_tier
    hdr.sender_model_hash = sender_model_hash
    hdr.receiver_model_hash = receiver_model_hash
    hdr.tokenizer_hash = tokenizer_hash
    hdr.raw_bytes = len(payload) if raw_bytes is None else raw_bytes
    hdr.sequence_id = sequence_id

    _, p_ptr = _as_uint8_ptr(payload)
    out_buf = (ctypes.c_uint8 * CMCP_HEADER_SIZE)()
    rc = _LIB.cmcp_header_encode(
        ctypes.byref(hdr),
        p_ptr if payload else None,
        len(payload),
        out_buf,
    )
    if rc != CMCP_OK:
        raise CMCPError(rc, f"cmcp_header_encode failed with status {rc}")
    return hdr, bytes(out_buf)


def cmcp_header_decode(header_bytes: bytes, payload: Optional[bytes] = None) -> CMCPHeader:
    if len(header_bytes) != CMCP_HEADER_SIZE:
        raise CMCPError(CMCP_ERR_INVALID_ARG, f"Expected {CMCP_HEADER_SIZE} header bytes, got {len(header_bytes)}")
    in_buf = (ctypes.c_uint8 * CMCP_HEADER_SIZE).from_buffer_copy(header_bytes)
    hdr = CMCPHeader()
    rc = _LIB.cmcp_header_decode(in_buf, ctypes.byref(hdr))
    if rc != CMCP_OK:
        raise CMCPError(rc, f"cmcp_header_decode failed with status {rc}")
    if payload is not None:
        _, p_ptr = _as_uint8_ptr(payload)
        vrc = _LIB.cmcp_verify_payload(ctypes.byref(hdr), p_ptr if payload else None, len(payload))
        if vrc != CMCP_OK:
            raise CMCPError(vrc, f"cmcp_verify_payload failed with status {vrc}")
    return hdr


def cmcp_tlv_write(records: Iterable[Tuple[int, bytes]]) -> bytes:
    rec_list = list(records)
    est_cap = sum(12 + len(val) for _, val in rec_list) + 64
    buf = (ctypes.c_uint8 * est_cap)()
    writer = CMCPTLVWriter()
    _LIB.cmcp_tlv_writer_init(ctypes.byref(writer), buf, est_cap)
    for tag, val in rec_list:
        v_buf = ctypes.create_string_buffer(val, max(1, len(val)))
        rc = _LIB.cmcp_tlv_write(
            ctypes.byref(writer),
            tag,
            0,
            ctypes.cast(v_buf, ctypes.c_void_p) if val else None,
            len(val),
        )
        if rc != CMCP_OK:
            raise CMCPError(rc, f"cmcp_tlv_write failed with status {rc}")
    return bytes(buf[: writer.pos])


def cmcp_tlv_read(data: bytes) -> List[Tuple[int, bytes]]:
    if not data:
        return []
    arr, ptr = _as_uint8_ptr(data)
    reader = CMCPTLVReader()
    _LIB.cmcp_tlv_reader_init(ctypes.byref(reader), ptr, len(data))
    out: List[Tuple[int, bytes]] = []
    rec = CMCPTLVRecord()
    while True:
        rc = _LIB.cmcp_tlv_next(ctypes.byref(reader), ctypes.byref(rec))
        if rc == 0:
            break
        if rc < 0:
            raise CMCPError(rc, f"cmcp_tlv_next failed with status {rc}")
        val = ctypes.string_at(rec.value, rec.length) if rec.length > 0 else b""
        out.append((int(rec.tag), val))
    return out


def cmcp_compress(data: bytes) -> bytes:
    bound = int(_LIB.cmcp_compress_bound(len(data)))
    dst = (ctypes.c_uint8 * max(1, bound))()
    out_len = ctypes.c_size_t(0)
    _, src_ptr = _as_uint8_ptr(data)
    rc = _LIB.cmcp_compress(
        src_ptr if data else None,
        len(data),
        dst,
        bound,
        ctypes.byref(out_len),
    )
    if rc != CMCP_OK:
        raise CMCPError(rc, f"cmcp_compress failed with status {rc}")
    return bytes(dst[: out_len.value])


def cmcp_decompress(data: bytes, raw_bytes: int) -> bytes:
    dst_cap = max(1, raw_bytes)
    dst = (ctypes.c_uint8 * dst_cap)()
    out_len = ctypes.c_size_t(0)
    _, src_ptr = _as_uint8_ptr(data)
    rc = _LIB.cmcp_decompress(
        src_ptr if data else None,
        len(data),
        dst,
        dst_cap,
        ctypes.byref(out_len),
    )
    if rc != CMCP_OK:
        raise CMCPError(rc, f"cmcp_decompress failed with status {rc}")
    return bytes(dst[: out_len.value])


def make_model_profile(
    arch_name: str,
    weights_hash: int = 0,
    tokenizer_hash: int = 0,
    n_layer: int = 0,
    n_embd: int = 0,
    n_head_kv: int = 0,
    rope_type: int = 0,
    rope_freq_scale: float = 1.0,
    kv_dtype: int = 1,
    n_ctx: int = 512,
    pos_offset: int = 0,
) -> CMCPModelProfile:
    prof = CMCPModelProfile()
    _LIB.cmcp_model_profile_init(ctypes.byref(prof), arch_name.encode("utf-8"))
    prof.weights_hash = weights_hash
    prof.tokenizer_hash = tokenizer_hash
    prof.n_layer = n_layer
    prof.n_embd = n_embd
    prof.n_head_kv = n_head_kv
    prof.rope_type = rope_type
    prof.rope_freq_scale = rope_freq_scale
    prof.kv_dtype = kv_dtype
    prof.n_ctx = n_ctx
    prof.pos_offset = pos_offset
    return prof


def cmcp_check_kv_compat(src: CMCPModelProfile, dst: CMCPModelProfile) -> CMCPCompatResult:
    res = CMCPCompatResult()
    _LIB.cmcp_check_kv_compat(ctypes.byref(src), ctypes.byref(dst), ctypes.byref(res))
    return res


def cmcp_check_compat(
    src: CMCPModelProfile,
    dst: CMCPModelProfile,
    strict_kv: bool = False,
) -> CMCPCompatResult:
    res = CMCPCompatResult()
    _LIB.cmcp_check_compat(
        ctypes.byref(src),
        ctypes.byref(dst),
        1 if strict_kv else 0,
        ctypes.byref(res),
    )
    return res


def encode_frame(
    records: Iterable[Tuple[int, bytes]],
    compress: bool = True,
    msg_type: int = CMCP_MSG_CONTEXT_TRANSFER,
    compat_tier: int = CMCP_TIER_2_CROSS_MODEL,
    sender_model_hash: int = 0,
    receiver_model_hash: int = 0,
    tokenizer_hash: int = 0,
    sequence_id: int = 1,
) -> bytes:
    raw_tlv = cmcp_tlv_write(records)
    raw_len = len(raw_tlv)
    flags = CMCP_FLAG_NONE
    payload = raw_tlv
    if compress and raw_len > 0:
        compressed = cmcp_compress(raw_tlv)
        if len(compressed) < raw_len:
            payload = compressed
            flags |= CMCP_FLAG_COMPRESSED

    _, hdr_bytes = cmcp_header_encode(
        payload=payload,
        msg_type=msg_type,
        compat_tier=compat_tier,
        flags=flags,
        sender_model_hash=sender_model_hash,
        receiver_model_hash=receiver_model_hash,
        tokenizer_hash=tokenizer_hash,
        raw_bytes=raw_len,
        sequence_id=sequence_id,
    )
    return hdr_bytes + payload


def decode_frame(frame_bytes: bytes) -> Tuple[CMCPHeader, List[Tuple[int, bytes]]]:
    if len(frame_bytes) < CMCP_HEADER_SIZE:
        raise CMCPError(CMCP_ERR_INVALID_ARG, "Frame shorter than 64-byte CMCP header")
    hdr_bytes = frame_bytes[:CMCP_HEADER_SIZE]
    payload = frame_bytes[CMCP_HEADER_SIZE:]
    hdr = cmcp_header_decode(hdr_bytes, payload=payload)
    if hdr.flags & CMCP_FLAG_COMPRESSED:
        raw_tlv = cmcp_decompress(payload, hdr.raw_bytes)
    else:
        raw_tlv = payload
    records = cmcp_tlv_read(raw_tlv)
    return hdr, records
