#include "cmcp/cmcp.h"
#include <string.h>

static void write_u16_le(uint8_t *dst, uint16_t v) {
    dst[0] = (uint8_t)(v & 0xFFu);
    dst[1] = (uint8_t)((v >> 8) & 0xFFu);
}

static void write_u32_le(uint8_t *dst, uint32_t v) {
    dst[0] = (uint8_t)(v & 0xFFu);
    dst[1] = (uint8_t)((v >> 8) & 0xFFu);
    dst[2] = (uint8_t)((v >> 16) & 0xFFu);
    dst[3] = (uint8_t)((v >> 24) & 0xFFu);
}

static void write_u64_le(uint8_t *dst, uint64_t v) {
    dst[0] = (uint8_t)(v & 0xFFu);
    dst[1] = (uint8_t)((v >> 8) & 0xFFu);
    dst[2] = (uint8_t)((v >> 16) & 0xFFu);
    dst[3] = (uint8_t)((v >> 24) & 0xFFu);
    dst[4] = (uint8_t)((v >> 32) & 0xFFu);
    dst[5] = (uint8_t)((v >> 40) & 0xFFu);
    dst[6] = (uint8_t)((v >> 48) & 0xFFu);
    dst[7] = (uint8_t)((v >> 56) & 0xFFu);
}

static uint16_t read_u16_le(const uint8_t *src) {
    return (uint16_t)((uint16_t)src[0] | ((uint16_t)src[1] << 8));
}

static uint32_t read_u32_le(const uint8_t *src) {
    return ((uint32_t)src[0]) |
           ((uint32_t)src[1] << 8) |
           ((uint32_t)src[2] << 16) |
           ((uint32_t)src[3] << 24);
}

static uint64_t read_u64_le(const uint8_t *src) {
    return ((uint64_t)src[0]) |
           ((uint64_t)src[1] << 8) |
           ((uint64_t)src[2] << 16) |
           ((uint64_t)src[3] << 24) |
           ((uint64_t)src[4] << 32) |
           ((uint64_t)src[5] << 40) |
           ((uint64_t)src[6] << 48) |
           ((uint64_t)src[7] << 56);
}

void cmcp_header_init(cmcp_header_t *hdr) {
    if (hdr == NULL) {
        return;
    }
    memset(hdr, 0, sizeof(*hdr));
    hdr->magic[0] = CMCP_MAGIC_0;
    hdr->magic[1] = CMCP_MAGIC_1;
    hdr->magic[2] = CMCP_MAGIC_2;
    hdr->magic[3] = CMCP_MAGIC_3;
    hdr->version = CMCP_PROTOCOL_VERSION;
    hdr->compat_tier = CMCP_TIER_3_TEXT_FALLBACK;
}

cmcp_status_t cmcp_header_encode(
    cmcp_header_t *hdr,
    const uint8_t *payload,
    size_t         payload_len,
    uint8_t        out_buf[CMCP_HEADER_SIZE]) {

    uint32_t payload_crc = 0u;
    uint32_t header_crc  = 0u;

    if (hdr == NULL || out_buf == NULL) {
        return CMCP_ERR_INVALID_ARG;
    }
    if (payload_len > 0 && payload == NULL) {
        return CMCP_ERR_INVALID_ARG;
    }
    if (payload_len > CMCP_MAX_PAYLOAD_BYTES) {
        return CMCP_ERR_LIMIT_EXCEEDED;
    }

    hdr->magic[0] = CMCP_MAGIC_0;
    hdr->magic[1] = CMCP_MAGIC_1;
    hdr->magic[2] = CMCP_MAGIC_2;
    hdr->magic[3] = CMCP_MAGIC_3;
    if (hdr->version == 0) {
        hdr->version = CMCP_PROTOCOL_VERSION;
    }
    if (hdr->version != CMCP_PROTOCOL_VERSION) {
        return CMCP_ERR_UNSUPPORTED_VERSION;
    }

    hdr->payload_bytes = (uint32_t)payload_len;
    if ((hdr->flags & CMCP_FLAG_COMPRESSED) == 0u) {
        hdr->raw_bytes = hdr->payload_bytes;
    } else if (hdr->raw_bytes > CMCP_MAX_PAYLOAD_BYTES) {
        return CMCP_ERR_LIMIT_EXCEEDED;
    }

    if (payload_len > 0) {
        payload_crc = cmcp_crc32c(0u, payload, payload_len);
    }
    hdr->payload_crc32c = payload_crc;
    hdr->reserved = 0u;

    out_buf[0] = hdr->magic[0];
    out_buf[1] = hdr->magic[1];
    out_buf[2] = hdr->magic[2];
    out_buf[3] = hdr->magic[3];
    write_u16_le(out_buf + 4,  hdr->version);
    write_u16_le(out_buf + 6,  hdr->flags);
    write_u16_le(out_buf + 8,  hdr->msg_type);
    write_u16_le(out_buf + 10, hdr->compat_tier);
    write_u64_le(out_buf + 12, hdr->sender_model_hash);
    write_u64_le(out_buf + 20, hdr->receiver_model_hash);
    write_u64_le(out_buf + 28, hdr->tokenizer_hash);
    write_u32_le(out_buf + 36, hdr->payload_bytes);
    write_u32_le(out_buf + 40, hdr->raw_bytes);
    write_u64_le(out_buf + 44, hdr->sequence_id);
    write_u32_le(out_buf + 52, hdr->reserved);
    write_u32_le(out_buf + 56, hdr->payload_crc32c);

    header_crc = cmcp_crc32c(0u, out_buf, 60u);
    hdr->header_crc32c = header_crc;
    write_u32_le(out_buf + 60, hdr->header_crc32c);

    return CMCP_OK;
}

cmcp_status_t cmcp_header_decode(
    const uint8_t  in_buf[CMCP_HEADER_SIZE],
    cmcp_header_t *out_hdr) {

    uint32_t expected_crc;
    uint32_t actual_crc;

    if (in_buf == NULL || out_hdr == NULL) {
        return CMCP_ERR_INVALID_ARG;
    }

    if (in_buf[0] != CMCP_MAGIC_0 ||
        in_buf[1] != CMCP_MAGIC_1 ||
        in_buf[2] != CMCP_MAGIC_2 ||
        in_buf[3] != CMCP_MAGIC_3) {
        return CMCP_ERR_BAD_MAGIC;
    }

    expected_crc = read_u32_le(in_buf + 60);
    actual_crc   = cmcp_crc32c(0u, in_buf, 60u);
    if (expected_crc != actual_crc) {
        return CMCP_ERR_HEADER_CRC;
    }

    memset(out_hdr, 0, sizeof(*out_hdr));
    out_hdr->magic[0] = in_buf[0];
    out_hdr->magic[1] = in_buf[1];
    out_hdr->magic[2] = in_buf[2];
    out_hdr->magic[3] = in_buf[3];
    out_hdr->version             = read_u16_le(in_buf + 4);
    out_hdr->flags               = read_u16_le(in_buf + 6);
    out_hdr->msg_type            = read_u16_le(in_buf + 8);
    out_hdr->compat_tier         = read_u16_le(in_buf + 10);
    out_hdr->sender_model_hash   = read_u64_le(in_buf + 12);
    out_hdr->receiver_model_hash = read_u64_le(in_buf + 20);
    out_hdr->tokenizer_hash      = read_u64_le(in_buf + 28);
    out_hdr->payload_bytes       = read_u32_le(in_buf + 36);
    out_hdr->raw_bytes           = read_u32_le(in_buf + 40);
    out_hdr->sequence_id         = read_u64_le(in_buf + 44);
    out_hdr->reserved            = read_u32_le(in_buf + 52);
    out_hdr->payload_crc32c      = read_u32_le(in_buf + 56);
    out_hdr->header_crc32c       = expected_crc;

    if (out_hdr->version != CMCP_PROTOCOL_VERSION) {
        return CMCP_ERR_UNSUPPORTED_VERSION;
    }
    if (out_hdr->payload_bytes > CMCP_MAX_PAYLOAD_BYTES ||
        out_hdr->raw_bytes > CMCP_MAX_PAYLOAD_BYTES) {
        return CMCP_ERR_LIMIT_EXCEEDED;
    }

    return CMCP_OK;
}

cmcp_status_t cmcp_verify_payload(
    const cmcp_header_t *hdr,
    const uint8_t       *payload,
    size_t               payload_len) {

    uint32_t actual_crc = 0u;

    if (hdr == NULL) {
        return CMCP_ERR_INVALID_ARG;
    }
    if (payload_len != (size_t)hdr->payload_bytes) {
        return CMCP_ERR_CORRUPT_DATA;
    }
    if (payload_len > 0 && payload == NULL) {
        return CMCP_ERR_INVALID_ARG;
    }
    if (payload_len > 0) {
        actual_crc = cmcp_crc32c(0u, payload, payload_len);
    }
    if (actual_crc != hdr->payload_crc32c) {
        return CMCP_ERR_PAYLOAD_CRC;
    }
    return CMCP_OK;
}
