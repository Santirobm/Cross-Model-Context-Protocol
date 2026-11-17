#include "cmcp/cmcp.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#endif

struct cmcp_shm_channel {
    cmcp_channel_kind_t kind;
    size_t              capacity;
    uint8_t            *buf;
    size_t              frame_len;
    char                name[260];
#ifdef _WIN32
    HANDLE              shm_handle;
#endif
};

static void write_u32_le_chan(uint8_t *dst, uint32_t v) {
    dst[0] = (uint8_t)(v & 0xFFu);
    dst[1] = (uint8_t)((v >> 8) & 0xFFu);
    dst[2] = (uint8_t)((v >> 16) & 0xFFu);
    dst[3] = (uint8_t)((v >> 24) & 0xFFu);
}

static uint32_t read_u32_le_chan(const uint8_t *src) {
    return ((uint32_t)src[0]) |
           ((uint32_t)src[1] << 8) |
           ((uint32_t)src[2] << 16) |
           ((uint32_t)src[3] << 24);
}

static cmcp_status_t channel_alloc_common(
    cmcp_channel_kind_t   kind,
    const char           *name_or_path,
    size_t                capacity_bytes,
    int                   create_new,
    cmcp_shm_channel_t  **out_chan) {

    cmcp_shm_channel_t *ch;

    if (out_chan == NULL || capacity_bytes < CMCP_HEADER_SIZE) {
        return CMCP_ERR_INVALID_ARG;
    }
    if (capacity_bytes > (size_t)0xFFFFFFFFu - 4u) {
        return CMCP_ERR_LIMIT_EXCEEDED;
    }
    if ((kind == CMCP_CHANNEL_FILE || kind == CMCP_CHANNEL_SHM) &&
        (name_or_path == NULL || name_or_path[0] == '\0')) {
        return CMCP_ERR_INVALID_ARG;
    }

    ch = (cmcp_shm_channel_t *)calloc(1u, sizeof(*ch));
    if (ch == NULL) {
        return CMCP_ERR_IO;
    }
    ch->kind = kind;
    ch->capacity = capacity_bytes;

    if (name_or_path != NULL) {
        size_t nlen = strlen(name_or_path);
        if (nlen >= sizeof(ch->name)) {
            nlen = sizeof(ch->name) - 1u;
        }
        memcpy(ch->name, name_or_path, nlen);
        ch->name[nlen] = '\0';
    }

    if (kind == CMCP_CHANNEL_MEMORY) {
        ch->buf = (uint8_t *)calloc(1u, capacity_bytes);
        if (ch->buf == NULL) {
            free(ch);
            return CMCP_ERR_IO;
        }
    } else if (kind == CMCP_CHANNEL_FILE) {
        ch->buf = (uint8_t *)calloc(1u, capacity_bytes);
        if (ch->buf == NULL) {
            free(ch);
            return CMCP_ERR_IO;
        }
        if (create_new) {
            FILE *fp = fopen(ch->name, "wb");
            if (fp == NULL) {
                free(ch->buf);
                free(ch);
                return CMCP_ERR_IO;
            }
            fclose(fp);
        }
    } else if (kind == CMCP_CHANNEL_SHM) {
#ifdef _WIN32
        DWORD total_size = (DWORD)(capacity_bytes + 4u);
        if (create_new) {
            ch->shm_handle = CreateFileMappingA(
                INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0u, total_size, ch->name);
        } else {
            ch->shm_handle = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, ch->name);
        }
        if (ch->shm_handle == NULL) {
            free(ch);
            return CMCP_ERR_IO;
        }
        ch->buf = (uint8_t *)MapViewOfFile(
            ch->shm_handle, FILE_MAP_ALL_ACCESS, 0u, 0u, (SIZE_T)total_size);
        if (ch->buf == NULL) {
            CloseHandle(ch->shm_handle);
            free(ch);
            return CMCP_ERR_IO;
        }
#else
        /* Portable fallback for POSIX without non-C99 extensions */
        ch->buf = (uint8_t *)calloc(1u, capacity_bytes + 4u);
        if (ch->buf == NULL) {
            free(ch);
            return CMCP_ERR_IO;
        }
#endif
    } else {
        free(ch);
        return CMCP_ERR_INVALID_ARG;
    }

    *out_chan = ch;
    return CMCP_OK;
}

cmcp_status_t cmcp_shm_channel_create(
    cmcp_channel_kind_t    kind,
    const char            *name_or_path,
    size_t                 capacity_bytes,
    cmcp_shm_channel_t   **out_chan) {
    return channel_alloc_common(kind, name_or_path, capacity_bytes, 1, out_chan);
}

cmcp_status_t cmcp_shm_channel_open(
    cmcp_channel_kind_t    kind,
    const char            *name_or_path,
    size_t                 capacity_bytes,
    cmcp_shm_channel_t   **out_chan) {
    return channel_alloc_common(kind, name_or_path, capacity_bytes, 0, out_chan);
}

cmcp_status_t cmcp_shm_channel_send_frame(
    cmcp_shm_channel_t  *chan,
    const cmcp_header_t *hdr,
    const uint8_t       *payload,
    size_t               payload_len) {

    cmcp_header_t local_hdr;
    uint8_t       hdr_bytes[CMCP_HEADER_SIZE];
    size_t        total_len;
    cmcp_status_t st;

    if (chan == NULL || hdr == NULL) {
        return CMCP_ERR_INVALID_ARG;
    }
    if (payload_len > 0u && payload == NULL) {
        return CMCP_ERR_INVALID_ARG;
    }

    if (payload_len > CMCP_MAX_PAYLOAD_BYTES) {
        return CMCP_ERR_LIMIT_EXCEEDED;
    }
    if (payload_len > chan->capacity - CMCP_HEADER_SIZE) {
        return CMCP_ERR_BUFFER_TOO_SMALL;
    }
    total_len = CMCP_HEADER_SIZE + payload_len;

    local_hdr = *hdr;
    st = cmcp_header_encode(&local_hdr, payload, payload_len, hdr_bytes);
    if (st != CMCP_OK) {
        return st;
    }

    if (chan->kind == CMCP_CHANNEL_MEMORY) {
        memcpy(chan->buf, hdr_bytes, CMCP_HEADER_SIZE);
        if (payload_len > 0u) {
            memcpy(chan->buf + CMCP_HEADER_SIZE, payload, payload_len);
        }
        chan->frame_len = total_len;
    } else if (chan->kind == CMCP_CHANNEL_FILE) {
        FILE *fp = fopen(chan->name, "wb");
        if (fp == NULL) {
            return CMCP_ERR_IO;
        }
        if (fwrite(hdr_bytes, 1u, CMCP_HEADER_SIZE, fp) != CMCP_HEADER_SIZE ||
            (payload_len > 0u && fwrite(payload, 1u, payload_len, fp) != payload_len)) {
            fclose(fp);
            return CMCP_ERR_IO;
        }
        fclose(fp);
        chan->frame_len = total_len;
    } else if (chan->kind == CMCP_CHANNEL_SHM) {
        write_u32_le_chan(chan->buf, (uint32_t)total_len);
        memcpy(chan->buf + 4u, hdr_bytes, CMCP_HEADER_SIZE);
        if (payload_len > 0u) {
            memcpy(chan->buf + 4u + CMCP_HEADER_SIZE, payload, payload_len);
        }
        chan->frame_len = total_len;
    }

    return CMCP_OK;
}

cmcp_status_t cmcp_shm_channel_recv_frame(
    cmcp_shm_channel_t *chan,
    cmcp_header_t      *out_hdr,
    uint8_t            *out_payload,
    size_t              payload_cap,
    size_t             *out_payload_len) {

    const uint8_t *frame_ptr = NULL;
    size_t         frame_len = 0u;
    cmcp_status_t  st;

    if (chan == NULL || out_hdr == NULL || out_payload_len == NULL) {
        return CMCP_ERR_INVALID_ARG;
    }

    if (chan->kind == CMCP_CHANNEL_MEMORY) {
        if (chan->frame_len < CMCP_HEADER_SIZE) {
            return CMCP_ERR_IO;
        }
        frame_ptr = chan->buf;
        frame_len = chan->frame_len;
    } else if (chan->kind == CMCP_CHANNEL_FILE) {
        FILE *fp = fopen(chan->name, "rb");
        size_t nread;
        if (fp == NULL) {
            return CMCP_ERR_IO;
        }
        nread = fread(chan->buf, 1u, chan->capacity, fp);
        fclose(fp);
        if (nread < CMCP_HEADER_SIZE) {
            return CMCP_ERR_IO;
        }
        frame_ptr = chan->buf;
        frame_len = nread;
    } else if (chan->kind == CMCP_CHANNEL_SHM) {
        uint32_t stored_len = read_u32_le_chan(chan->buf);
        if (stored_len < CMCP_HEADER_SIZE || (size_t)stored_len > chan->capacity) {
            return CMCP_ERR_CORRUPT_DATA;
        }
        frame_ptr = chan->buf + 4u;
        frame_len = (size_t)stored_len;
    } else {
        return CMCP_ERR_INVALID_ARG;
    }

    st = cmcp_header_decode(frame_ptr, out_hdr);
    if (st != CMCP_OK) {
        return st;
    }

    if (CMCP_HEADER_SIZE + (size_t)out_hdr->payload_bytes > frame_len) {
        return CMCP_ERR_CORRUPT_DATA;
    }
    if ((size_t)out_hdr->payload_bytes > payload_cap) {
        return CMCP_ERR_BUFFER_TOO_SMALL;
    }
    if (out_hdr->payload_bytes > 0u && out_payload == NULL) {
        return CMCP_ERR_INVALID_ARG;
    }

    st = cmcp_verify_payload(
        out_hdr,
        frame_ptr + CMCP_HEADER_SIZE,
        (size_t)out_hdr->payload_bytes);
    if (st != CMCP_OK) {
        return st;
    }

    if (out_hdr->payload_bytes > 0u) {
        memcpy(out_payload, frame_ptr + CMCP_HEADER_SIZE, (size_t)out_hdr->payload_bytes);
    }
    *out_payload_len = (size_t)out_hdr->payload_bytes;
    return CMCP_OK;
}

void cmcp_shm_channel_close(cmcp_shm_channel_t *chan) {
    if (chan == NULL) {
        return;
    }
#ifdef _WIN32
    if (chan->kind == CMCP_CHANNEL_SHM) {
        if (chan->buf != NULL) {
            UnmapViewOfFile(chan->buf);
            chan->buf = NULL;
        }
        if (chan->shm_handle != NULL) {
            CloseHandle(chan->shm_handle);
            chan->shm_handle = NULL;
        }
    }
#endif
    if (chan->buf != NULL) {
        free(chan->buf);
        chan->buf = NULL;
    }
    free(chan);
}
