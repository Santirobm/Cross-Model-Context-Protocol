#ifndef CMCP_H
#define CMCP_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
#  if defined(CMCP_SHARED)
#    if defined(CMCP_BUILD_SHARED)
#      define CMCP_API __declspec(dllexport)
#    else
#      define CMCP_API __declspec(dllimport)
#    endif
#  else
#    define CMCP_API
#  endif
#else
#  if defined(CMCP_SHARED) && defined(__GNUC__) && __GNUC__ >= 4
#    define CMCP_API __attribute__((visibility("default")))
#  else
#    define CMCP_API
#  endif
#endif

/* ========================================================================= */
/* Constants & Error Codes                                                   */
/* ========================================================================= */

#define CMCP_MAGIC_0 'C'
#define CMCP_MAGIC_1 'M'
#define CMCP_MAGIC_2 'C'
#define CMCP_MAGIC_3 'P'

#define CMCP_PROTOCOL_VERSION 1u
#define CMCP_HEADER_SIZE      64u
#define CMCP_MAX_PAYLOAD_BYTES (64u * 1024u * 1024u)

/* Header flags */
#define CMCP_FLAG_NONE        0x0000u
#define CMCP_FLAG_COMPRESSED  0x0001u
#define CMCP_FLAG_HAS_KV_DESC 0x0002u
#define CMCP_FLAG_STRICT_KV   0x0004u

/* Status / error codes */
typedef enum cmcp_status {
    CMCP_OK                      =  0,
    CMCP_ERR_INVALID_ARG         = -1,
    CMCP_ERR_BUFFER_TOO_SMALL    = -2,
    CMCP_ERR_BAD_MAGIC           = -3,
    CMCP_ERR_UNSUPPORTED_VERSION = -4,
    CMCP_ERR_HEADER_CRC          = -5,
    CMCP_ERR_PAYLOAD_CRC         = -6,
    CMCP_ERR_CORRUPT_DATA        = -7,
    CMCP_ERR_INCOMPATIBLE        = -8,
    CMCP_ERR_IO                  = -9,
    CMCP_ERR_LIMIT_EXCEEDED      = -10
} cmcp_status_t;

/* Message types */
typedef enum cmcp_msg_type {
    CMCP_MSG_HELLO            = 0x0001u,
    CMCP_MSG_CONTEXT_TRANSFER = 0x0002u,
    CMCP_MSG_KV_SNAPSHOT      = 0x0003u,
    CMCP_MSG_LATENT_TRANSFER  = 0x0004u,
    CMCP_MSG_ACK              = 0x0005u,
    CMCP_MSG_REJECT           = 0x0006u
} cmcp_msg_type_t;

/* 4-tier compatibility levels */
typedef enum cmcp_compat_tier {
    CMCP_TIER_0_EXACT_KV      = 0x0000u,
    CMCP_TIER_1_SAME_FAMILY   = 0x0001u,
    CMCP_TIER_2_CROSS_MODEL   = 0x0002u,
    CMCP_TIER_3_TEXT_FALLBACK = 0x0003u,
    CMCP_TIER_INCOMPATIBLE    = 0xFFFFu
} cmcp_compat_tier_t;

/* TLV record tags */
typedef enum cmcp_tlv_tag {
    CMCP_TLV_CONCEPT        = 0x01u,
    CMCP_TLV_ENTITY         = 0x02u,
    CMCP_TLV_RELATION       = 0x03u,
    CMCP_TLV_INSTRUCTION    = 0x04u,
    CMCP_TLV_CONSTRAINT     = 0x05u,
    CMCP_TLV_EVIDENCE       = 0x06u,
    CMCP_TLV_REFERENCE      = 0x07u,
    CMCP_TLV_DECISION       = 0x08u,
    CMCP_TLV_LATENT_TENSOR  = 0x09u,
    CMCP_TLV_KV_COMPAT_DESC = 0x0Au
} cmcp_tlv_tag_t;

/* Compatibility mismatch reason codes */
typedef enum cmcp_compat_reason {
    CMCP_COMPAT_OK                   = 0,
    CMCP_COMPAT_MISMATCH_ARCH        = 1,
    CMCP_COMPAT_MISMATCH_WEIGHTS     = 2,
    CMCP_COMPAT_MISMATCH_TOKENIZER   = 3,
    CMCP_COMPAT_MISMATCH_GEOMETRY    = 4,
    CMCP_COMPAT_MISMATCH_ROPE        = 5,
    CMCP_COMPAT_MISMATCH_KV_DTYPE    = 6,
    CMCP_COMPAT_MISMATCH_CTX         = 7,
    CMCP_COMPAT_MISMATCH_POS_OFFSET  = 8,
    CMCP_COMPAT_INVALID_PROFILE      = 9
} cmcp_compat_reason_t;

/* ========================================================================= */
/* Data Structures                                                           */
/* ========================================================================= */

typedef struct cmcp_header {
    uint8_t  magic[4];
    uint16_t version;
    uint16_t flags;
    uint16_t msg_type;
    uint16_t compat_tier;
    uint64_t sender_model_hash;
    uint64_t receiver_model_hash;
    uint64_t tokenizer_hash;
    uint32_t payload_bytes;
    uint32_t raw_bytes;
    uint64_t sequence_id;
    uint32_t reserved;
    uint32_t payload_crc32c;
    uint32_t header_crc32c;
} cmcp_header_t;

typedef struct cmcp_tlv_record {
    uint8_t        tag;
    uint8_t        flags;
    uint32_t       length;
    const uint8_t *value;
} cmcp_tlv_record_t;

typedef struct cmcp_tlv_writer {
    uint8_t *buf;
    size_t   cap;
    size_t   pos;
} cmcp_tlv_writer_t;

typedef struct cmcp_tlv_reader {
    const uint8_t *buf;
    size_t         len;
    size_t         pos;
} cmcp_tlv_reader_t;

typedef struct cmcp_model_profile {
    char     arch_name[64];
    uint64_t arch_hash;
    uint64_t weights_hash;
    uint64_t tokenizer_hash;
    uint32_t n_layer;
    uint32_t n_embd;
    uint32_t n_head_kv;
    int32_t  rope_type;
    float    rope_freq_scale;
    uint32_t kv_dtype;
    uint32_t n_ctx;
    uint32_t pos_offset;
} cmcp_model_profile_t;

typedef struct cmcp_compat_result {
    cmcp_compat_tier_t   tier;
    int                  kv_compatible;
    cmcp_compat_reason_t reason_code;
    char                 reason[128];
} cmcp_compat_result_t;

typedef enum cmcp_channel_kind {
    CMCP_CHANNEL_MEMORY = 0,
    CMCP_CHANNEL_FILE   = 1,
    CMCP_CHANNEL_SHM    = 2
} cmcp_channel_kind_t;

typedef struct cmcp_shm_channel cmcp_shm_channel_t;

/* ========================================================================= */
/* CRC32C (Castagnoli) & Hashing                                             */
/* ========================================================================= */

CMCP_API uint32_t cmcp_crc32c(uint32_t init_crc, const void *data, size_t len);
CMCP_API uint64_t cmcp_fnv1a_64(const void *data, size_t len);

/* ========================================================================= */
/* 64-Byte Binary Header Encode / Decode                                     */
/* ========================================================================= */

CMCP_API void cmcp_header_init(cmcp_header_t *hdr);

CMCP_API cmcp_status_t cmcp_header_encode(
    cmcp_header_t *hdr,
    const uint8_t *payload,
    size_t         payload_len,
    uint8_t        out_buf[CMCP_HEADER_SIZE]);

CMCP_API cmcp_status_t cmcp_header_decode(
    const uint8_t  in_buf[CMCP_HEADER_SIZE],
    cmcp_header_t *out_hdr);

CMCP_API cmcp_status_t cmcp_verify_payload(
    const cmcp_header_t *hdr,
    const uint8_t       *payload,
    size_t               payload_len);

/* ========================================================================= */
/* TLV Serialization & Varint + RLE/LZ-lite Lossless Compression             */
/* ========================================================================= */

CMCP_API void cmcp_tlv_writer_init(cmcp_tlv_writer_t *w, uint8_t *buf, size_t cap);

CMCP_API cmcp_status_t cmcp_tlv_write(
    cmcp_tlv_writer_t *w,
    uint8_t            tag,
    uint8_t            flags,
    const void        *value,
    uint32_t           length);

CMCP_API void cmcp_tlv_reader_init(cmcp_tlv_reader_t *r, const uint8_t *buf, size_t len);

/* Returns 1 if a record was read, 0 at end-of-buffer, or < 0 (cmcp_status_t) on error. */
CMCP_API int cmcp_tlv_next(cmcp_tlv_reader_t *r, cmcp_tlv_record_t *out_rec);

CMCP_API size_t cmcp_compress_bound(size_t src_len);

CMCP_API cmcp_status_t cmcp_compress(
    const uint8_t *src,
    size_t         src_len,
    uint8_t       *dst,
    size_t         dst_cap,
    size_t        *out_len);

CMCP_API cmcp_status_t cmcp_decompress(
    const uint8_t *src,
    size_t         src_len,
    uint8_t       *dst,
    size_t         dst_cap,
    size_t        *out_len);

/* ========================================================================= */
/* Compatibility & KV-Cache Validation                                       */
/* ========================================================================= */

CMCP_API void cmcp_model_profile_init(
    cmcp_model_profile_t *profile,
    const char           *arch_name);

CMCP_API cmcp_status_t cmcp_check_kv_compat(
    const cmcp_model_profile_t *src,
    const cmcp_model_profile_t *dst,
    cmcp_compat_result_t       *out);

CMCP_API cmcp_status_t cmcp_check_compat(
    const cmcp_model_profile_t *src,
    const cmcp_model_profile_t *dst,
    int                         strict_kv,
    cmcp_compat_result_t       *out);

/* ========================================================================= */
/* Lightweight Memory / File / SHM Transport Channel                         */
/* ========================================================================= */

CMCP_API cmcp_status_t cmcp_shm_channel_create(
    cmcp_channel_kind_t    kind,
    const char            *name_or_path,
    size_t                 capacity_bytes,
    cmcp_shm_channel_t   **out_chan);

CMCP_API cmcp_status_t cmcp_shm_channel_open(
    cmcp_channel_kind_t    kind,
    const char            *name_or_path,
    size_t                 capacity_bytes,
    cmcp_shm_channel_t   **out_chan);

CMCP_API cmcp_status_t cmcp_shm_channel_send_frame(
    cmcp_shm_channel_t  *chan,
    const cmcp_header_t *hdr,
    const uint8_t       *payload,
    size_t               payload_len);

CMCP_API cmcp_status_t cmcp_shm_channel_recv_frame(
    cmcp_shm_channel_t *chan,
    cmcp_header_t      *out_hdr,
    uint8_t            *out_payload,
    size_t              payload_cap,
    size_t             *out_payload_len);

CMCP_API void cmcp_shm_channel_close(cmcp_shm_channel_t *chan);

#ifdef __cplusplus
}
#endif

#endif /* CMCP_H */
