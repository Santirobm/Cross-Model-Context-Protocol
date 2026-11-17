#include "cmcp/cmcp.h"

/* Castagnoli CRC32C reflected polynomial: 0x82F63B78 */
#define CMCP_CRC32C_POLY 0x82F63B78u

static uint32_t g_crc32c_table[256];
static int      g_crc32c_init = 0;

static void cmcp_crc32c_init_table(void) {
    uint32_t i;
    if (g_crc32c_init) {
        return;
    }
    for (i = 0; i < 256u; ++i) {
        uint32_t crc = i;
        int bit;
        for (bit = 0; bit < 8; ++bit) {
            if (crc & 1u) {
                crc = (crc >> 1) ^ CMCP_CRC32C_POLY;
            } else {
                crc >>= 1;
            }
        }
        g_crc32c_table[i] = crc;
    }
    g_crc32c_init = 1;
}

uint32_t cmcp_crc32c(uint32_t init_crc, const void *data, size_t len) {
    const uint8_t *bytes = (const uint8_t *)data;
    uint32_t crc;
    size_t i;

    if (!g_crc32c_init) {
        cmcp_crc32c_init_table();
    }
    if (data == NULL && len > 0) {
        return init_crc;
    }

    crc = init_crc ^ 0xFFFFFFFFu;
    for (i = 0; i < len; ++i) {
        uint8_t idx = (uint8_t)((crc ^ bytes[i]) & 0xFFu);
        crc = (crc >> 8) ^ g_crc32c_table[idx];
    }
    return crc ^ 0xFFFFFFFFu;
}

uint64_t cmcp_fnv1a_64(const void *data, size_t len) {
    const uint8_t *bytes = (const uint8_t *)data;
    uint64_t hash = 14695981039346656037ULL;
    size_t i;

    if (data == NULL && len > 0) {
        return 0;
    }
    for (i = 0; i < len; ++i) {
        hash ^= (uint64_t)bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}
