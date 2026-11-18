#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#include "cmcp/cmcp.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

static void set_reason(cmcp_compat_result_t *out, cmcp_compat_reason_t code, const char *msg) {
    out->reason_code = code;
    if (msg != NULL) {
        size_t len = strlen(msg);
        if (len >= sizeof(out->reason)) {
            len = sizeof(out->reason) - 1u;
        }
        memcpy(out->reason, msg, len);
        out->reason[len] = '\0';
    } else {
        out->reason[0] = '\0';
    }
}

void cmcp_model_profile_init(cmcp_model_profile_t *profile, const char *arch_name) {
    if (profile == NULL) {
        return;
    }
    memset(profile, 0, sizeof(*profile));
    if (arch_name != NULL) {
        size_t len = strlen(arch_name);
        if (len >= sizeof(profile->arch_name)) {
            len = sizeof(profile->arch_name) - 1u;
        }
        memcpy(profile->arch_name, arch_name, len);
        profile->arch_name[len] = '\0';
        profile->arch_hash = cmcp_fnv1a_64(arch_name, len);
    }
    profile->rope_freq_scale = 1.0f;
}

cmcp_status_t cmcp_fingerprint_file(const char *path, uint64_t *out_hash) {
    FILE *fp;
    uint8_t buf[65536];
    uint64_t h = 14695981039346656037ULL;
    uint64_t file_size = 0u;
    size_t n_head;
    size_t i;

    if (path == NULL || out_hash == NULL || path[0] == '\0') {
        return CMCP_ERR_INVALID_ARG;
    }

    fp = fopen(path, "rb");
    if (fp == NULL) {
        return CMCP_ERR_IO;
    }

    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return CMCP_ERR_IO;
    }
    {
        long sz = ftell(fp);
        if (sz < 0) {
            fclose(fp);
            return CMCP_ERR_IO;
        }
        file_size = (uint64_t)sz;
    }
    if (fseek(fp, 0, SEEK_SET) != 0) {
        fclose(fp);
        return CMCP_ERR_IO;
    }

    for (i = 0u; i < 8u; ++i) {
        h ^= (uint8_t)((file_size >> (i * 8u)) & 0xFFu);
        h *= 1099511628211ULL;
    }

    n_head = fread(buf, 1u, sizeof(buf), fp);
    for (i = 0u; i < n_head; ++i) {
        h ^= (uint64_t)buf[i];
        h *= 1099511628211ULL;
    }

    if (file_size > (uint64_t)sizeof(buf)) {
        uint64_t tail_off = file_size - (uint64_t)sizeof(buf);
        if (fseek(fp, (long)tail_off, SEEK_SET) == 0) {
            size_t n_tail = fread(buf, 1u, sizeof(buf), fp);
            for (i = 0u; i < n_tail; ++i) {
                h ^= (uint64_t)buf[i];
                h *= 1099511628211ULL;
            }
        }
    }

    fclose(fp);
    *out_hash = h;
    return CMCP_OK;
}

cmcp_status_t cmcp_check_kv_compat(
    const cmcp_model_profile_t *src,
    const cmcp_model_profile_t *dst,
    cmcp_compat_result_t       *out) {

    float scale_diff;

    if (src == NULL || dst == NULL || out == NULL) {
        return CMCP_ERR_INVALID_ARG;
    }

    memset(out, 0, sizeof(*out));
    out->tier = CMCP_TIER_INCOMPATIBLE;
    out->kv_compatible = 0;

    if (src->arch_hash == 0u || dst->arch_hash == 0u ||
        src->n_layer == 0u || dst->n_layer == 0u ||
        src->n_embd == 0u || dst->n_embd == 0u) {
        set_reason(out, CMCP_COMPAT_INVALID_PROFILE, "incomplete model profile");
        return CMCP_ERR_INCOMPATIBLE;
    }

    if (src->arch_hash != dst->arch_hash ||
        strcmp(src->arch_name, dst->arch_name) != 0) {
        set_reason(out, CMCP_COMPAT_MISMATCH_ARCH, "architecture mismatch");
        return CMCP_ERR_INCOMPATIBLE;
    }

    if (src->weights_hash != dst->weights_hash) {
        set_reason(out, CMCP_COMPAT_MISMATCH_WEIGHTS, "weights hash mismatch");
        return CMCP_ERR_INCOMPATIBLE;
    }

    if (src->tokenizer_hash != dst->tokenizer_hash) {
        set_reason(out, CMCP_COMPAT_MISMATCH_TOKENIZER, "tokenizer hash mismatch");
        return CMCP_ERR_INCOMPATIBLE;
    }

    if (src->n_layer != dst->n_layer ||
        src->n_embd != dst->n_embd ||
        src->n_head_kv != dst->n_head_kv) {
        set_reason(out, CMCP_COMPAT_MISMATCH_GEOMETRY, "layer/embd/head_kv geometry mismatch");
        return CMCP_ERR_INCOMPATIBLE;
    }

    scale_diff = src->rope_freq_scale - dst->rope_freq_scale;
    if (scale_diff < 0.0f) {
        scale_diff = -scale_diff;
    }
    if (src->rope_type != dst->rope_type || scale_diff > 1e-6f) {
        set_reason(out, CMCP_COMPAT_MISMATCH_ROPE, "RoPE type or freq_scale mismatch");
        return CMCP_ERR_INCOMPATIBLE;
    }

    if (src->kv_dtype != dst->kv_dtype) {
        set_reason(out, CMCP_COMPAT_MISMATCH_KV_DTYPE, "KV cache dtype mismatch");
        return CMCP_ERR_INCOMPATIBLE;
    }

    if (src->n_ctx != dst->n_ctx) {
        set_reason(out, CMCP_COMPAT_MISMATCH_CTX, "n_ctx mismatch");
        return CMCP_ERR_INCOMPATIBLE;
    }

    if (src->pos_offset != dst->pos_offset) {
        set_reason(out, CMCP_COMPAT_MISMATCH_POS_OFFSET, "pos_offset mismatch");
        return CMCP_ERR_INCOMPATIBLE;
    }

    out->tier = CMCP_TIER_0_EXACT_KV;
    out->kv_compatible = 1;
    set_reason(out, CMCP_COMPAT_OK, "exact KV-cache match (Tier 0)");
    return CMCP_OK;
}

cmcp_status_t cmcp_check_compat(
    const cmcp_model_profile_t *src,
    const cmcp_model_profile_t *dst,
    int                         strict_kv,
    cmcp_compat_result_t       *out) {

    cmcp_status_t kv_status;

    if (src == NULL || dst == NULL || out == NULL) {
        return CMCP_ERR_INVALID_ARG;
    }

    kv_status = cmcp_check_kv_compat(src, dst, out);
    if (kv_status == CMCP_OK) {
        return CMCP_OK;
    }

    if (strict_kv) {
        out->tier = CMCP_TIER_INCOMPATIBLE;
        out->kv_compatible = 0;
        return CMCP_ERR_INCOMPATIBLE;
    }

    if (out->reason_code == CMCP_COMPAT_INVALID_PROFILE) {
        out->tier = CMCP_TIER_INCOMPATIBLE;
        return CMCP_ERR_INCOMPATIBLE;
    }

    if (src->arch_hash == dst->arch_hash &&
        strcmp(src->arch_name, dst->arch_name) == 0 &&
        src->n_embd == dst->n_embd) {
        out->tier = CMCP_TIER_1_SAME_FAMILY;
        set_reason(out, out->reason_code, "same-family model (Tier 1: projection/re-prefill)");
        return CMCP_OK;
    }

    if (src->n_embd > 0u && dst->n_embd > 0u) {
        out->tier = CMCP_TIER_2_CROSS_MODEL;
        set_reason(out, out->reason_code, "cross-model pair (Tier 2: latent alignment / TLV)");
        return CMCP_OK;
    }

    out->tier = CMCP_TIER_3_TEXT_FALLBACK;
    set_reason(out, out->reason_code, "text fallback (Tier 3: portable TLV)");
    return CMCP_OK;
}
