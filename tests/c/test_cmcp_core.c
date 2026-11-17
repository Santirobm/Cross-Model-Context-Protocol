#include "cmcp/cmcp.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_CHECK(cond)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "TEST FAILED at %s:%d: %s\n",                      \
                    __FILE__, __LINE__, #cond);                                \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

static void test_crc32c(void) {
    /* Standard Castagnoli CRC32C check value for "123456789" is 0xE3069283 */
    const char *vec = "123456789";
    uint32_t crc = cmcp_crc32c(0u, vec, 9u);
    TEST_CHECK(crc == 0xE3069283u);

    /* Incremental update */
    uint32_t c1 = cmcp_crc32c(0u, vec, 4u);
    uint32_t c2 = cmcp_crc32c(c1, vec + 4, 5u);
    TEST_CHECK(c2 == 0xE3069283u);
    printf("[PASS] test_crc32c\n");
}

static void test_header_roundtrip_and_corruption(void) {
    const uint8_t payload[] = "Hello CMCP binary envelope!";
    uint8_t wire[CMCP_HEADER_SIZE];
    cmcp_header_t hdr;
    cmcp_header_t decoded;
    cmcp_status_t st;

    cmcp_header_init(&hdr);
    hdr.msg_type = CMCP_MSG_CONTEXT_TRANSFER;
    hdr.compat_tier = CMCP_TIER_2_CROSS_MODEL;
    hdr.sender_model_hash = 0x1122334455667788ULL;
    hdr.receiver_model_hash = 0x8877665544332211ULL;
    hdr.tokenizer_hash = 0xCAFEBABEDEADBEEFULL;
    hdr.sequence_id = 42ULL;

    st = cmcp_header_encode(&hdr, payload, sizeof(payload) - 1u, wire);
    TEST_CHECK(st == CMCP_OK);

    st = cmcp_header_decode(wire, &decoded);
    TEST_CHECK(st == CMCP_OK);
    TEST_CHECK(decoded.version == CMCP_PROTOCOL_VERSION);
    TEST_CHECK(decoded.msg_type == CMCP_MSG_CONTEXT_TRANSFER);
    TEST_CHECK(decoded.compat_tier == CMCP_TIER_2_CROSS_MODEL);
    TEST_CHECK(decoded.sender_model_hash == 0x1122334455667788ULL);
    TEST_CHECK(decoded.receiver_model_hash == 0x8877665544332211ULL);
    TEST_CHECK(decoded.tokenizer_hash == 0xCAFEBABEDEADBEEFULL);
    TEST_CHECK(decoded.payload_bytes == (uint32_t)(sizeof(payload) - 1u));
    TEST_CHECK(decoded.raw_bytes == decoded.payload_bytes);
    TEST_CHECK(decoded.sequence_id == 42ULL);

    st = cmcp_verify_payload(&decoded, payload, sizeof(payload) - 1u);
    TEST_CHECK(st == CMCP_OK);

    /* Corrupt a header byte -> must fail header CRC */
    wire[15] ^= 0x01u;
    st = cmcp_header_decode(wire, &decoded);
    TEST_CHECK(st == CMCP_ERR_HEADER_CRC);
    wire[15] ^= 0x01u;

    /* Corrupt magic -> must fail bad magic */
    wire[0] = 'X';
    st = cmcp_header_decode(wire, &decoded);
    TEST_CHECK(st == CMCP_ERR_BAD_MAGIC);
    wire[0] = 'C';

    /* Corrupt payload -> must fail payload CRC */
    {
        uint8_t bad_payload[sizeof(payload)];
        memcpy(bad_payload, payload, sizeof(payload));
        bad_payload[3] ^= 0x80u;
        st = cmcp_header_decode(wire, &decoded);
        TEST_CHECK(st == CMCP_OK);
        st = cmcp_verify_payload(&decoded, bad_payload, sizeof(payload) - 1u);
        TEST_CHECK(st == CMCP_ERR_PAYLOAD_CRC);
    }

    printf("[PASS] test_header_roundtrip_and_corruption\n");
}

static void test_tlv_and_compression(void) {
    uint8_t tlv_buf[2048];
    cmcp_tlv_writer_t w;
    cmcp_tlv_reader_t r;
    cmcp_tlv_record_t rec;
    int rc;

    const char *inst = "Summarize the architectural constraints of the protocol.";
    const char *evid = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
    const char *decs = "Use Tier 2 cross-model latent transfer with TLV fallback.";

    cmcp_tlv_writer_init(&w, tlv_buf, sizeof(tlv_buf));
    TEST_CHECK(cmcp_tlv_write(&w, CMCP_TLV_INSTRUCTION, 0u, inst, (uint32_t)strlen(inst)) == CMCP_OK);
    TEST_CHECK(cmcp_tlv_write(&w, CMCP_TLV_EVIDENCE, 0u, evid, (uint32_t)strlen(evid)) == CMCP_OK);
    TEST_CHECK(cmcp_tlv_write(&w, CMCP_TLV_DECISION, 1u, decs, (uint32_t)strlen(decs)) == CMCP_OK);
    TEST_CHECK(cmcp_tlv_write(&w, CMCP_TLV_CONCEPT, 0u, "CMCP", 4u) == CMCP_OK);

    /* Compress TLV buffer */
    {
        size_t bound = cmcp_compress_bound(w.pos);
        uint8_t *comp = (uint8_t *)malloc(bound);
        uint8_t *decomp = (uint8_t *)malloc(w.pos);
        size_t comp_len = 0u;
        size_t decomp_len = 0u;

        TEST_CHECK(comp != NULL && decomp != NULL);
        TEST_CHECK(cmcp_compress(tlv_buf, w.pos, comp, bound, &comp_len) == CMCP_OK);
        TEST_CHECK(comp_len < w.pos); /* RLE + LZ-lite should shrink repeated 'A's */

        TEST_CHECK(cmcp_decompress(comp, comp_len, decomp, w.pos, &decomp_len) == CMCP_OK);
        TEST_CHECK(decomp_len == w.pos);
        TEST_CHECK(memcmp(decomp, tlv_buf, w.pos) == 0);

        /* Zip-bomb / too-small dst_cap check */
        TEST_CHECK(cmcp_decompress(comp, comp_len, decomp, w.pos - 1u, &decomp_len) ==
               CMCP_ERR_BUFFER_TOO_SMALL);

        /* Truncated compressed stream check */
        TEST_CHECK(cmcp_decompress(comp, comp_len - 2u, decomp, w.pos, &decomp_len) ==
               CMCP_ERR_CORRUPT_DATA);

        /* Read back TLV records from decompressed buffer */
        cmcp_tlv_reader_init(&r, decomp, decomp_len);
        rc = cmcp_tlv_next(&r, &rec);
        TEST_CHECK(rc == 1 && rec.tag == CMCP_TLV_INSTRUCTION && rec.length == strlen(inst));
        TEST_CHECK(memcmp(rec.value, inst, rec.length) == 0);

        rc = cmcp_tlv_next(&r, &rec);
        TEST_CHECK(rc == 1 && rec.tag == CMCP_TLV_EVIDENCE && rec.length == strlen(evid));

        rc = cmcp_tlv_next(&r, &rec);
        TEST_CHECK(rc == 1 && rec.tag == CMCP_TLV_DECISION && rec.flags == 1u);

        rc = cmcp_tlv_next(&r, &rec);
        TEST_CHECK(rc == 1 && rec.tag == CMCP_TLV_CONCEPT && rec.length == 4u);

        rc = cmcp_tlv_next(&r, &rec);
        TEST_CHECK(rc == 0);

        /* Truncated TLV check */
        cmcp_tlv_reader_init(&r, decomp, decomp_len - 1u);
        TEST_CHECK(cmcp_tlv_next(&r, &rec) == 1);
        TEST_CHECK(cmcp_tlv_next(&r, &rec) == 1);
        TEST_CHECK(cmcp_tlv_next(&r, &rec) == 1);
        TEST_CHECK(cmcp_tlv_next(&r, &rec) == (int)CMCP_ERR_CORRUPT_DATA);

        free(comp);
        free(decomp);
    }

    printf("[PASS] test_tlv_and_compression\n");
}

static void test_compat_and_kv_validation(void) {
    cmcp_model_profile_t qwen_a;
    cmcp_model_profile_t qwen_b;
    cmcp_model_profile_t llama_b;
    cmcp_compat_result_t res;

    cmcp_model_profile_init(&qwen_a, "qwen2");
    qwen_a.weights_hash = 0xAAA1u;
    qwen_a.tokenizer_hash = 0xBBB1u;
    qwen_a.n_layer = 24u;
    qwen_a.n_embd = 896u;
    qwen_a.n_head_kv = 2u;
    qwen_a.rope_type = 2;
    qwen_a.rope_freq_scale = 1.0f;
    qwen_a.kv_dtype = 1u;
    qwen_a.n_ctx = 512u;
    qwen_a.pos_offset = 0u;

    qwen_b = qwen_a;
    TEST_CHECK(cmcp_check_kv_compat(&qwen_a, &qwen_b, &res) == CMCP_OK);
    TEST_CHECK(res.tier == CMCP_TIER_0_EXACT_KV);
    TEST_CHECK(res.kv_compatible == 1);

    /* Different n_ctx -> KV must reject, non-strict negotiates Tier 1 */
    qwen_b.n_ctx = 1024u;
    TEST_CHECK(cmcp_check_kv_compat(&qwen_a, &qwen_b, &res) == CMCP_ERR_INCOMPATIBLE);
    TEST_CHECK(res.reason_code == CMCP_COMPAT_MISMATCH_CTX);
    TEST_CHECK(cmcp_check_compat(&qwen_a, &qwen_b, 1, &res) == CMCP_ERR_INCOMPATIBLE);
    TEST_CHECK(cmcp_check_compat(&qwen_a, &qwen_b, 0, &res) == CMCP_OK);
    TEST_CHECK(res.tier == CMCP_TIER_1_SAME_FAMILY);

    /* Different pos_offset -> KV must reject */
    qwen_b = qwen_a;
    qwen_b.pos_offset = 16u;
    TEST_CHECK(cmcp_check_kv_compat(&qwen_a, &qwen_b, &res) == CMCP_ERR_INCOMPATIBLE);
    TEST_CHECK(res.reason_code == CMCP_COMPAT_MISMATCH_POS_OFFSET);

    /* Different architecture -> KV rejects, non-strict negotiates Tier 2 */
    cmcp_model_profile_init(&llama_b, "llama");
    llama_b.weights_hash = 0xCCC1u;
    llama_b.tokenizer_hash = 0xDDD1u;
    llama_b.n_layer = 16u;
    llama_b.n_embd = 2048u;
    llama_b.n_head_kv = 8u;
    llama_b.rope_type = 0;
    llama_b.rope_freq_scale = 1.0f;
    llama_b.kv_dtype = 1u;
    llama_b.n_ctx = 512u;
    llama_b.pos_offset = 0u;

    TEST_CHECK(cmcp_check_kv_compat(&qwen_a, &llama_b, &res) == CMCP_ERR_INCOMPATIBLE);
    TEST_CHECK(res.reason_code == CMCP_COMPAT_MISMATCH_ARCH);
    TEST_CHECK(cmcp_check_compat(&qwen_a, &llama_b, 0, &res) == CMCP_OK);
    TEST_CHECK(res.tier == CMCP_TIER_2_CROSS_MODEL);

    printf("[PASS] test_compat_and_kv_validation\n");
}

static void test_shm_channel(void) {
    cmcp_shm_channel_t *tx = NULL;
    cmcp_shm_channel_t *rx = NULL;
    cmcp_header_t hdr;
    cmcp_header_t rx_hdr;
    const uint8_t msg[] = "SHM_FRAME_PAYLOAD_TEST";
    uint8_t rx_buf[128];
    size_t rx_len = 0u;

    cmcp_header_init(&hdr);
    hdr.msg_type = CMCP_MSG_CONTEXT_TRANSFER;
    hdr.sequence_id = 7u;

    /* In-memory channel */
    TEST_CHECK(cmcp_shm_channel_create(CMCP_CHANNEL_MEMORY, NULL, 1024u, &tx) == CMCP_OK);
    TEST_CHECK(cmcp_shm_channel_send_frame(tx, &hdr, msg, sizeof(msg) - 1u) == CMCP_OK);
    TEST_CHECK(cmcp_shm_channel_recv_frame(tx, &rx_hdr, rx_buf, sizeof(rx_buf), &rx_len) == CMCP_OK);
    TEST_CHECK(rx_len == sizeof(msg) - 1u);
    TEST_CHECK(memcmp(rx_buf, msg, rx_len) == 0);
    TEST_CHECK(rx_hdr.sequence_id == 7u);
    cmcp_shm_channel_close(tx);

    /* Named SHM channel (Windows CreateFileMappingA / OpenFileMappingA) */
    TEST_CHECK(cmcp_shm_channel_create(CMCP_CHANNEL_SHM, "Local\\CMCP_Test_SHM_1", 1024u, &tx) == CMCP_OK);
    TEST_CHECK(cmcp_shm_channel_open(CMCP_CHANNEL_SHM, "Local\\CMCP_Test_SHM_1", 1024u, &rx) == CMCP_OK);
    TEST_CHECK(cmcp_shm_channel_send_frame(tx, &hdr, msg, sizeof(msg) - 1u) == CMCP_OK);
    TEST_CHECK(cmcp_shm_channel_recv_frame(rx, &rx_hdr, rx_buf, sizeof(rx_buf), &rx_len) == CMCP_OK);
    TEST_CHECK(rx_len == sizeof(msg) - 1u);
    TEST_CHECK(memcmp(rx_buf, msg, rx_len) == 0);
    cmcp_shm_channel_close(rx);
    cmcp_shm_channel_close(tx);

    /* Overflow / limit checks */
    TEST_CHECK(cmcp_shm_channel_create(CMCP_CHANNEL_MEMORY, NULL, (size_t)0xFFFFFFFFu, &tx) ==
               CMCP_ERR_LIMIT_EXCEEDED);
    TEST_CHECK(cmcp_shm_channel_create(CMCP_CHANNEL_MEMORY, NULL, 128u, &tx) == CMCP_OK);
    TEST_CHECK(cmcp_shm_channel_send_frame(tx, &hdr, msg, (size_t)-1) == CMCP_ERR_LIMIT_EXCEEDED);
    cmcp_shm_channel_close(tx);

    printf("[PASS] test_shm_channel\n");
}

int main(void) {
    test_crc32c();
    test_header_roundtrip_and_corruption();
    test_tlv_and_compression();
    test_compat_and_kv_validation();
    test_shm_channel();
    printf("All CMCP core C99 tests passed.\n");
    return 0;
}
