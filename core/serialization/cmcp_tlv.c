#include "cmcp/cmcp.h"
#include <string.h>

/* ========================================================================= */
/* ULEB128 Varint Helpers                                                    */
/* ========================================================================= */

static size_t uleb128_encode_u64(uint64_t val, uint8_t *dst) {
    size_t n = 0u;
    do {
        uint8_t byte = (uint8_t)(val & 0x7Fu);
        val >>= 7;
        if (val != 0u) {
            byte |= 0x80u;
        }
        dst[n++] = byte;
    } while (val != 0u);
    return n;
}

static cmcp_status_t uleb128_decode_u64(
    const uint8_t *src,
    size_t         src_len,
    size_t        *io_pos,
    uint64_t      *out_val) {

    uint64_t result = 0u;
    unsigned shift = 0u;
    size_t pos = *io_pos;

    while (pos < src_len) {
        uint8_t byte = src[pos++];
        if (shift >= 64u || (shift == 63u && (byte & 0x7Fu) > 1u)) {
            return CMCP_ERR_CORRUPT_DATA;
        }
        result |= ((uint64_t)(byte & 0x7Fu)) << shift;
        if ((byte & 0x80u) == 0u) {
            *io_pos = pos;
            *out_val = result;
            return CMCP_OK;
        }
        shift += 7u;
    }
    return CMCP_ERR_CORRUPT_DATA;
}

/* ========================================================================= */
/* TLV Writer & Reader                                                       */
/* ========================================================================= */

void cmcp_tlv_writer_init(cmcp_tlv_writer_t *w, uint8_t *buf, size_t cap) {
    if (w == NULL) {
        return;
    }
    w->buf = buf;
    w->cap = cap;
    w->pos = 0u;
}

cmcp_status_t cmcp_tlv_write(
    cmcp_tlv_writer_t *w,
    uint8_t            tag,
    uint8_t            flags,
    const void        *value,
    uint32_t           length) {

    uint8_t len_buf[5];
    size_t  len_bytes;
    size_t  needed;

    if (w == NULL || w->buf == NULL) {
        return CMCP_ERR_INVALID_ARG;
    }
    if (length > 0u && value == NULL) {
        return CMCP_ERR_INVALID_ARG;
    }

    len_bytes = uleb128_encode_u64((uint64_t)length, len_buf);
    needed = 2u + len_bytes + (size_t)length;

    if (w->pos > w->cap || needed > w->cap - w->pos) {
        return CMCP_ERR_BUFFER_TOO_SMALL;
    }

    w->buf[w->pos++] = tag;
    w->buf[w->pos++] = flags;
    memcpy(w->buf + w->pos, len_buf, len_bytes);
    w->pos += len_bytes;
    if (length > 0u) {
        memcpy(w->buf + w->pos, value, (size_t)length);
        w->pos += (size_t)length;
    }
    return CMCP_OK;
}

void cmcp_tlv_reader_init(cmcp_tlv_reader_t *r, const uint8_t *buf, size_t len) {
    if (r == NULL) {
        return;
    }
    r->buf = buf;
    r->len = len;
    r->pos = 0u;
}

int cmcp_tlv_next(cmcp_tlv_reader_t *r, cmcp_tlv_record_t *out_rec) {
    uint64_t rec_len = 0u;
    size_t   cursor;

    if (r == NULL || out_rec == NULL) {
        return (int)CMCP_ERR_INVALID_ARG;
    }
    if (r->pos == r->len) {
        return 0;
    }
    if (r->buf == NULL || r->pos > r->len || r->len - r->pos < 3u) {
        return (int)CMCP_ERR_CORRUPT_DATA;
    }

    cursor = r->pos;
    out_rec->tag   = r->buf[cursor++];
    out_rec->flags = r->buf[cursor++];

    if (uleb128_decode_u64(r->buf, r->len, &cursor, &rec_len) != CMCP_OK) {
        return (int)CMCP_ERR_CORRUPT_DATA;
    }
    if (rec_len > 0xFFFFFFFFULL || rec_len > (uint64_t)(r->len - cursor)) {
        return (int)CMCP_ERR_CORRUPT_DATA;
    }

    out_rec->length = (uint32_t)rec_len;
    out_rec->value  = (rec_len > 0u) ? (r->buf + cursor) : NULL;
    r->pos = cursor + (size_t)rec_len;
    return 1;
}

/* ========================================================================= */
/* Lossless Varint + RLE / LZ-lite Compression                               */
/* ========================================================================= */

#define CMCP_LZ_HASH_BITS 10u
#define CMCP_LZ_HASH_SIZE (1u << CMCP_LZ_HASH_BITS)
#define CMCP_LZ_MAX_DIST  65535u

static uint32_t lz_hash4(const uint8_t *p) {
    uint32_t v = ((uint32_t)p[0]) |
                 ((uint32_t)p[1] << 8) |
                 ((uint32_t)p[2] << 16) |
                 ((uint32_t)p[3] << 24);
    return (v * 2654435761u) >> (32u - CMCP_LZ_HASH_BITS);
}

static cmcp_status_t flush_literals(
    const uint8_t *lit_start,
    size_t         lit_len,
    uint8_t       *dst,
    size_t         dst_cap,
    size_t        *io_out_pos) {

    size_t offset = 0u;
    while (offset < lit_len) {
        size_t chunk = lit_len - offset;
        if (chunk > 64u) {
            chunk = 64u;
        }
        if (*io_out_pos + 1u + chunk > dst_cap) {
            return CMCP_ERR_BUFFER_TOO_SMALL;
        }
        dst[(*io_out_pos)++] = (uint8_t)(chunk - 1u);
        memcpy(dst + *io_out_pos, lit_start + offset, chunk);
        *io_out_pos += chunk;
        offset += chunk;
    }
    return CMCP_OK;
}

size_t cmcp_compress_bound(size_t src_len) {
    return 10u + src_len + (src_len / 64u) + 8u;
}

cmcp_status_t cmcp_compress(
    const uint8_t *src,
    size_t         src_len,
    uint8_t       *dst,
    size_t         dst_cap,
    size_t        *out_len) {

    uint32_t htab[CMCP_LZ_HASH_SIZE];
    uint8_t  hdr_buf[10];
    size_t   hdr_len;
    size_t   out_pos = 0u;
    size_t   in_pos  = 0u;
    size_t   lit_anchor = 0u;
    size_t   i;

    if (dst == NULL || out_len == NULL || (src_len > 0u && src == NULL)) {
        return CMCP_ERR_INVALID_ARG;
    }
    if (src_len > CMCP_MAX_PAYLOAD_BYTES) {
        return CMCP_ERR_LIMIT_EXCEEDED;
    }

    hdr_len = uleb128_encode_u64((uint64_t)src_len, hdr_buf);
    if (hdr_len > dst_cap) {
        return CMCP_ERR_BUFFER_TOO_SMALL;
    }
    memcpy(dst, hdr_buf, hdr_len);
    out_pos = hdr_len;

    for (i = 0u; i < CMCP_LZ_HASH_SIZE; ++i) {
        htab[i] = 0xFFFFFFFFu;
    }

    while (in_pos < src_len) {
        size_t rle_len = 1u;
        size_t max_rle = src_len - in_pos;
        if (max_rle > 66u) {
            max_rle = 66u;
        }
        while (rle_len < max_rle && src[in_pos + rle_len] == src[in_pos]) {
            ++rle_len;
        }
        if (rle_len >= 3u) {
            cmcp_status_t st = flush_literals(
                src + lit_anchor, in_pos - lit_anchor, dst, dst_cap, &out_pos);
            if (st != CMCP_OK) {
                return st;
            }
            if (out_pos + 2u > dst_cap) {
                return CMCP_ERR_BUFFER_TOO_SMALL;
            }
            dst[out_pos++] = (uint8_t)(0x40u | (uint8_t)(rle_len - 3u));
            dst[out_pos++] = src[in_pos];
            in_pos += rle_len;
            lit_anchor = in_pos;
            continue;
        }

        if (src_len - in_pos >= 4u) {
            uint32_t h = lz_hash4(src + in_pos);
            uint32_t cand = htab[h];
            htab[h] = (uint32_t)in_pos;

            if (cand != 0xFFFFFFFFu && (size_t)cand < in_pos) {
                size_t dist = in_pos - (size_t)cand;
                if (dist >= 1u && dist <= CMCP_LZ_MAX_DIST) {
                    size_t match_len = 0u;
                    size_t max_match = src_len - in_pos;
                    if (max_match > 131u) {
                        max_match = 131u;
                    }
                    while (match_len < max_match &&
                           src[(size_t)cand + match_len] == src[in_pos + match_len]) {
                        ++match_len;
                    }
                    if (match_len >= 4u) {
                        uint8_t dist_buf[5];
                        size_t  dist_bytes;
                        cmcp_status_t st = flush_literals(
                            src + lit_anchor, in_pos - lit_anchor, dst, dst_cap, &out_pos);
                        if (st != CMCP_OK) {
                            return st;
                        }
                        dist_bytes = uleb128_encode_u64((uint64_t)dist, dist_buf);
                        if (out_pos + 1u + dist_bytes > dst_cap) {
                            return CMCP_ERR_BUFFER_TOO_SMALL;
                        }
                        dst[out_pos++] = (uint8_t)(0x80u | (uint8_t)(match_len - 4u));
                        memcpy(dst + out_pos, dist_buf, dist_bytes);
                        out_pos += dist_bytes;
                        in_pos += match_len;
                        lit_anchor = in_pos;
                        continue;
                    }
                }
            }
        }
        ++in_pos;
    }

    {
        cmcp_status_t st = flush_literals(
            src + lit_anchor, src_len - lit_anchor, dst, dst_cap, &out_pos);
        if (st != CMCP_OK) {
            return st;
        }
    }

    *out_len = out_pos;
    return CMCP_OK;
}

cmcp_status_t cmcp_decompress(
    const uint8_t *src,
    size_t         src_len,
    uint8_t       *dst,
    size_t         dst_cap,
    size_t        *out_len) {

    uint64_t raw_len_u64 = 0u;
    size_t   raw_len = 0u;
    size_t   in_pos  = 0u;
    size_t   out_pos = 0u;

    if (src == NULL || out_len == NULL || (dst_cap > 0u && dst == NULL)) {
        return CMCP_ERR_INVALID_ARG;
    }

    if (uleb128_decode_u64(src, src_len, &in_pos, &raw_len_u64) != CMCP_OK) {
        return CMCP_ERR_CORRUPT_DATA;
    }
    if (raw_len_u64 > (uint64_t)CMCP_MAX_PAYLOAD_BYTES) {
        return CMCP_ERR_LIMIT_EXCEEDED;
    }
    raw_len = (size_t)raw_len_u64;
    if (raw_len > dst_cap) {
        return CMCP_ERR_BUFFER_TOO_SMALL;
    }
    if (raw_len > 0u && dst == NULL) {
        return CMCP_ERR_INVALID_ARG;
    }

    while (in_pos < src_len && out_pos < raw_len) {
        uint8_t op = src[in_pos++];
        if ((op & 0xC0u) == 0x00u) {
            size_t count = (size_t)(op & 0x3Fu) + 1u;
            if (count > src_len - in_pos || count > raw_len - out_pos) {
                return CMCP_ERR_CORRUPT_DATA;
            }
            memcpy(dst + out_pos, src + in_pos, count);
            in_pos += count;
            out_pos += count;
        } else if ((op & 0xC0u) == 0x40u) {
            size_t count = (size_t)(op & 0x3Fu) + 3u;
            uint8_t val;
            if (in_pos >= src_len || count > raw_len - out_pos) {
                return CMCP_ERR_CORRUPT_DATA;
            }
            val = src[in_pos++];
            memset(dst + out_pos, (int)val, count);
            out_pos += count;
        } else {
            size_t   match_len = (size_t)(op & 0x7Fu) + 4u;
            uint64_t dist_u64  = 0u;
            size_t   dist;
            size_t   i;
            if (uleb128_decode_u64(src, src_len, &in_pos, &dist_u64) != CMCP_OK) {
                return CMCP_ERR_CORRUPT_DATA;
            }
            if (dist_u64 == 0u || dist_u64 > (uint64_t)out_pos || match_len > raw_len - out_pos) {
                return CMCP_ERR_CORRUPT_DATA;
            }
            dist = (size_t)dist_u64;
            for (i = 0u; i < match_len; ++i) {
                dst[out_pos + i] = dst[out_pos - dist + i];
            }
            out_pos += match_len;
        }
    }

    if (out_pos != raw_len || in_pos != src_len) {
        return CMCP_ERR_CORRUPT_DATA;
    }

    *out_len = out_pos;
    return CMCP_OK;
}
