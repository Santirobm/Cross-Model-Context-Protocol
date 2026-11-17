#include "cmcp/cmcp.h"
#include "llama.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#  include <psapi.h>
#endif

static void quiet_log_cb(enum ggml_log_level level, const char *text, void *user_data) {
    (void)user_data;
    if (level == GGML_LOG_LEVEL_ERROR) {
        fprintf(stderr, "%s", text);
    }
}

static double get_rss_mib(void) {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return (double)pmc.WorkingSetSize / (1024.0 * 1024.0);
    }
#endif
    return 0.0;
}

static void print_json_escaped(const char *s) {
    putchar('"');
    if (s != NULL) {
        for (; *s != '\0'; ++s) {
            unsigned char c = (unsigned char)*s;
            switch (c) {
                case '"':  printf("\\\""); break;
                case '\\': printf("\\\\"); break;
                case '\b': printf("\\b");  break;
                case '\f': printf("\\f");  break;
                case '\n': printf("\\n");  break;
                case '\r': printf("\\r");  break;
                case '\t': printf("\\t");  break;
                default:
                    if (c < 0x20u) {
                        printf("\\u%04x", (unsigned int)c);
                    } else {
                        putchar((int)c);
                    }
                    break;
            }
        }
    }
    putchar('"');
}

static void fill_profile_from_llama(
    const struct llama_model *model,
    uint32_t                  n_ctx,
    uint32_t                  pos_offset,
    cmcp_model_profile_t     *out) {

    char arch[64] = "unknown";
    char name[128] = "unknown";
    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    int32_t n_vocab = llama_vocab_n_tokens(vocab);

    llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch));
    llama_model_meta_val_str(model, "general.name", name, sizeof(name));

    cmcp_model_profile_init(out, arch);
    out->weights_hash = cmcp_fnv1a_64(name, strlen(name)) ^
                        ((uint64_t)llama_model_n_params(model) * 11400714819323198485ULL);
    out->tokenizer_hash = ((uint64_t)(uint32_t)n_vocab) ^ (cmcp_fnv1a_64(arch, strlen(arch)) << 1);
    out->n_layer = (uint32_t)llama_model_n_layer(model);
    out->n_embd = (uint32_t)llama_model_n_embd(model);
    out->n_head_kv = (uint32_t)llama_model_n_head_kv(model);
    out->rope_type = (int32_t)llama_model_rope_type(model);
    out->rope_freq_scale = llama_model_rope_freq_scale_train(model);
    out->kv_dtype = 1u; /* F16 default in llama_context_default_params */
    out->n_ctx = n_ctx;
    out->pos_offset = pos_offset;
}

static int tokenize_prompt(
    const struct llama_vocab *vocab,
    const char               *prompt,
    llama_token             **out_tokens,
    int32_t                  *out_n) {

    int32_t prompt_len = (int32_t)strlen(prompt);
    int32_t cap = prompt_len + 16;
    llama_token *toks = (llama_token *)malloc((size_t)cap * sizeof(llama_token));
    int32_t n;

    if (toks == NULL) {
        return -1;
    }
    n = llama_tokenize(vocab, prompt, prompt_len, toks, cap, true, true);
    if (n < 0) {
        cap = -n + 4;
        free(toks);
        toks = (llama_token *)malloc((size_t)cap * sizeof(llama_token));
        if (toks == NULL) {
            return -1;
        }
        n = llama_tokenize(vocab, prompt, prompt_len, toks, cap, true, true);
    }
    if (n < 0) {
        free(toks);
        return -1;
    }
    *out_tokens = toks;
    *out_n = n;
    return 0;
}

static int run_bench(
    const char *model_path,
    const char *prompt,
    int32_t     max_tokens,
    uint32_t    n_ctx,
    int32_t     n_threads) {

    struct llama_model_params mparams = llama_model_default_params();
    struct llama_model *model = llama_model_load_from_file(model_path, mparams);
    struct llama_context_params cparams;
    struct llama_context *ctx;
    const struct llama_vocab *vocab;
    cmcp_model_profile_t prof;

    llama_token *tokens = NULL;
    int32_t n_in = 0;
    int64_t t0, t1;
    double tokenize_ms, prefill_ms, eval_ms;

    char *out_text = NULL;
    size_t out_cap = 4096u;
    size_t out_len = 0u;
    int32_t n_gen = 0;
    struct llama_sampler *smpl;

    if (model == NULL) {
        fprintf(stderr, "Failed to load model: %s\n", model_path);
        return 1;
    }

    cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx;
    cparams.n_batch = n_ctx;
    cparams.n_threads = n_threads;
    cparams.n_threads_batch = n_threads;

    ctx = llama_init_from_model(model, cparams);
    if (ctx == NULL) {
        llama_model_free(model);
        fprintf(stderr, "Failed to create context\n");
        return 1;
    }

    vocab = llama_model_get_vocab(model);
    fill_profile_from_llama(model, llama_n_ctx(ctx), 0u, &prof);

    t0 = llama_time_us();
    if (tokenize_prompt(vocab, prompt, &tokens, &n_in) != 0) {
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    t1 = llama_time_us();
    tokenize_ms = (double)(t1 - t0) / 1000.0;

    t0 = llama_time_us();
    if (llama_decode(ctx, llama_batch_get_one(tokens, n_in)) != 0) {
        free(tokens);
        llama_free(ctx);
        llama_model_free(model);
        fprintf(stderr, "Prefill llama_decode failed\n");
        return 1;
    }
    t1 = llama_time_us();
    prefill_ms = (double)(t1 - t0) / 1000.0;

    out_text = (char *)calloc(1u, out_cap);
    smpl = llama_sampler_init_greedy();

    t0 = llama_time_us();
    for (int32_t i = 0; i < max_tokens; ++i) {
        llama_token next_tok = llama_sampler_sample(smpl, ctx, -1);
        char piece[256];
        int32_t plen;

        if (llama_vocab_is_eog(vocab, next_tok)) {
            break;
        }
        plen = llama_token_to_piece(vocab, next_tok, piece, (int32_t)sizeof(piece), 0, true);
        if (plen > 0) {
            if (out_len + (size_t)plen + 1u > out_cap) {
                out_cap = (out_len + (size_t)plen + 1u) * 2u;
                out_text = (char *)realloc(out_text, out_cap);
            }
            memcpy(out_text + out_len, piece, (size_t)plen);
            out_len += (size_t)plen;
            out_text[out_len] = '\0';
        }
        ++n_gen;
        if (llama_decode(ctx, llama_batch_get_one(&next_tok, 1)) != 0) {
            break;
        }
    }
    t1 = llama_time_us();
    eval_ms = (double)(t1 - t0) / 1000.0;

    printf("{\n");
    printf("  \"mode\": \"bench\",\n");
    printf("  \"model\": "); print_json_escaped(model_path); printf(",\n");
    printf("  \"arch\": "); print_json_escaped(prof.arch_name); printf(",\n");
    printf("  \"n_layer\": %u,\n", prof.n_layer);
    printf("  \"n_embd\": %u,\n", prof.n_embd);
    printf("  \"n_head_kv\": %u,\n", prof.n_head_kv);
    printf("  \"rope_type\": %d,\n", prof.rope_type);
    printf("  \"rope_freq_scale\": %.6f,\n", (double)prof.rope_freq_scale);
    printf("  \"n_ctx\": %u,\n", llama_n_ctx(ctx));
    printf("  \"n_input_tokens\": %d,\n", n_in);
    printf("  \"n_gen_tokens\": %d,\n", n_gen);
    printf("  \"tokenize_ms\": %.3f,\n", tokenize_ms);
    printf("  \"prefill_ms\": %.3f,\n", prefill_ms);
    printf("  \"eval_ms\": %.3f,\n", eval_ms);
    printf("  \"output_text\": "); print_json_escaped(out_text); printf(",\n");
    printf("  \"rss_mib\": %.2f\n", get_rss_mib());
    printf("}\n");

    llama_sampler_free(smpl);
    free(out_text);
    free(tokens);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}

static size_t test_kv_transfer(
    struct llama_context *src_ctx,
    struct llama_context *dst_ctx,
    const cmcp_model_profile_t *src_prof,
    const cmcp_model_profile_t *dst_prof,
    size_t *out_state_bytes) {

    size_t sz = llama_state_seq_get_size(src_ctx, 0);
    uint8_t *buf = (uint8_t *)malloc(sz);
    size_t got = 0u;
    size_t set_res = 0u;
    cmcp_compat_result_t cres;

    if (buf == NULL) {
        *out_state_bytes = 0u;
        return 0u;
    }
    got = llama_state_seq_get_data(src_ctx, buf, sz, 0);
    *out_state_bytes = got;
    set_res = llama_state_seq_set_data(dst_ctx, buf, got, 0);
    cmcp_check_kv_compat(src_prof, dst_prof, &cres);
    free(buf);
    return set_res;
}

static int run_kv_compat(
    const char *model_path,
    const char *other_model_path,
    const char *prompt,
    uint32_t    n_ctx,
    int32_t     n_threads) {

    struct llama_model_params mparams = llama_model_default_params();
    struct llama_model *m1 = llama_model_load_from_file(model_path, mparams);
    struct llama_model *m2 = NULL;
    struct llama_context_params cp1, cp_diff_ctx;
    struct llama_context *ctx_src = NULL;
    struct llama_context *ctx_same = NULL;
    struct llama_context *ctx_diff_ctx = NULL;
    struct llama_context *ctx_other = NULL;

    llama_token *tokens = NULL;
    int32_t n_in = 0;

    cmcp_model_profile_t p_src, p_same, p_diff_ctx, p_pos_off, p_other;
    cmcp_compat_result_t r_same, r_diff_ctx, r_pos_off, r_other;
    size_t bytes_a = 0, set_a = 0;
    size_t bytes_b1 = 0, set_b1 = 0;
    size_t bytes_c = 0, set_c = 0;

    if (m1 == NULL) {
        fprintf(stderr, "Failed to load primary model: %s\n", model_path);
        return 1;
    }
    if (other_model_path != NULL) {
        m2 = llama_model_load_from_file(other_model_path, mparams);
        if (m2 == NULL) {
            llama_model_free(m1);
            fprintf(stderr, "Failed to load other model: %s\n", other_model_path);
            return 1;
        }
    }

    cp1 = llama_context_default_params();
    cp1.n_ctx = n_ctx;
    cp1.n_batch = n_ctx;
    cp1.n_threads = n_threads;
    cp1.n_threads_batch = n_threads;

    ctx_src = llama_init_from_model(m1, cp1);
    ctx_same = llama_init_from_model(m1, cp1);

    cp_diff_ctx = cp1;
    cp_diff_ctx.n_ctx = n_ctx / 2u;
    if (cp_diff_ctx.n_ctx < 64u) {
        cp_diff_ctx.n_ctx = n_ctx * 2u;
    }
    ctx_diff_ctx = llama_init_from_model(m1, cp_diff_ctx);

    if (tokenize_prompt(llama_model_get_vocab(m1), prompt, &tokens, &n_in) != 0) {
        return 1;
    }
    if (llama_decode(ctx_src, llama_batch_get_one(tokens, n_in)) != 0) {
        return 1;
    }

    fill_profile_from_llama(m1, llama_n_ctx(ctx_src), 0u, &p_src);
    fill_profile_from_llama(m1, llama_n_ctx(ctx_same), 0u, &p_same);
    fill_profile_from_llama(m1, llama_n_ctx(ctx_diff_ctx), 0u, &p_diff_ctx);
    p_pos_off = p_src;
    p_pos_off.pos_offset = 16u;

    /* (a) Same model + same config */
    set_a = test_kv_transfer(ctx_src, ctx_same, &p_src, &p_same, &bytes_a);
    cmcp_check_kv_compat(&p_src, &p_same, &r_same);

    /* (b1) Same model + different n_ctx */
    set_b1 = test_kv_transfer(ctx_src, ctx_diff_ctx, &p_src, &p_diff_ctx, &bytes_b1);
    cmcp_check_kv_compat(&p_src, &p_diff_ctx, &r_diff_ctx);

    /* (b2) Same model + different pos_offset (CMCP protocol check) */
    cmcp_check_kv_compat(&p_src, &p_pos_off, &r_pos_off);

    /* (c) Different architecture */
    if (m2 != NULL) {
        ctx_other = llama_init_from_model(m2, cp1);
        fill_profile_from_llama(m2, llama_n_ctx(ctx_other), 0u, &p_other);
        set_c = test_kv_transfer(ctx_src, ctx_other, &p_src, &p_other, &bytes_c);
        cmcp_check_kv_compat(&p_src, &p_other, &r_other);
    }

    printf("{\n");
    printf("  \"mode\": \"kv-compat\",\n");
    printf("  \"primary_model\": "); print_json_escaped(model_path); printf(",\n");
    printf("  \"primary_arch\": "); print_json_escaped(p_src.arch_name); printf(",\n");
    printf("  \"cases\": {\n");
    printf("    \"same_model_same_cfg\": {\n");
    printf("      \"kv_state_bytes\": %zu,\n", bytes_a);
    printf("      \"llama_state_seq_set_data_bytes\": %zu,\n", set_a);
    printf("      \"llama_accepted\": %s,\n", set_a > 0u ? "true" : "false");
    printf("      \"cmcp_kv_compatible\": %s,\n", r_same.kv_compatible ? "true" : "false");
    printf("      \"cmcp_reason\": "); print_json_escaped(r_same.reason); printf("\n");
    printf("    },\n");
    printf("    \"same_model_diff_n_ctx\": {\n");
    printf("      \"src_n_ctx\": %u,\n", p_src.n_ctx);
    printf("      \"dst_n_ctx\": %u,\n", p_diff_ctx.n_ctx);
    printf("      \"kv_state_bytes\": %zu,\n", bytes_b1);
    printf("      \"llama_state_seq_set_data_bytes\": %zu,\n", set_b1);
    printf("      \"llama_accepted\": %s,\n", set_b1 > 0u ? "true" : "false");
    printf("      \"cmcp_kv_compatible\": %s,\n", r_diff_ctx.kv_compatible ? "true" : "false");
    printf("      \"cmcp_reason\": "); print_json_escaped(r_diff_ctx.reason); printf("\n");
    printf("    },\n");
    printf("    \"same_model_diff_pos_offset\": {\n");
    printf("      \"src_pos_offset\": %u,\n", p_src.pos_offset);
    printf("      \"dst_pos_offset\": %u,\n", p_pos_off.pos_offset);
    printf("      \"cmcp_kv_compatible\": %s,\n", r_pos_off.kv_compatible ? "true" : "false");
    printf("      \"cmcp_reason\": "); print_json_escaped(r_pos_off.reason); printf("\n");
    printf("    }");
    if (m2 != NULL) {
        printf(",\n    \"cross_architecture\": {\n");
        printf("      \"other_model\": "); print_json_escaped(other_model_path); printf(",\n");
        printf("      \"other_arch\": "); print_json_escaped(p_other.arch_name); printf(",\n");
        printf("      \"kv_state_bytes\": %zu,\n", bytes_c);
        printf("      \"llama_state_seq_set_data_bytes\": %zu,\n", set_c);
        printf("      \"llama_accepted\": %s,\n", set_c > 0u ? "true" : "false");
        printf("      \"cmcp_kv_compatible\": %s,\n", r_other.kv_compatible ? "true" : "false");
        printf("      \"cmcp_reason\": "); print_json_escaped(r_other.reason); printf("\n");
        printf("    }\n");
    } else {
        printf("\n");
    }
    printf("  },\n");
    printf("  \"rss_mib\": %.2f\n", get_rss_mib());
    printf("}\n");

    free(tokens);
    if (ctx_other) llama_free(ctx_other);
    if (m2) llama_model_free(m2);
    llama_free(ctx_diff_ctx);
    llama_free(ctx_same);
    llama_free(ctx_src);
    llama_model_free(m1);
    return 0;
}

static int run_embed(
    const char *model_path,
    const char *prompt,
    uint32_t    n_ctx,
    int32_t     n_threads) {

    struct llama_model_params mparams = llama_model_default_params();
    struct llama_model *model = llama_model_load_from_file(model_path, mparams);
    struct llama_context_params cparams;
    struct llama_context *ctx;
    llama_token *tokens = NULL;
    int32_t n_in = 0;
    int32_t n_embd;
    int64_t t0, t1;
    double tokenize_ms, prefill_ms;
    float *embd = NULL;
    double norm_sq = 0.0;

    if (model == NULL) {
        fprintf(stderr, "Failed to load model: %s\n", model_path);
        return 1;
    }

    cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx;
    cparams.n_batch = n_ctx;
    cparams.n_threads = n_threads;
    cparams.n_threads_batch = n_threads;
    cparams.embeddings = true;

    ctx = llama_init_from_model(model, cparams);
    if (ctx == NULL) {
        llama_model_free(model);
        return 1;
    }

    t0 = llama_time_us();
    if (tokenize_prompt(llama_model_get_vocab(model), prompt, &tokens, &n_in) != 0) {
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    t1 = llama_time_us();
    tokenize_ms = (double)(t1 - t0) / 1000.0;

    t0 = llama_time_us();
    if (llama_decode(ctx, llama_batch_get_one(tokens, n_in)) != 0) {
        free(tokens);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    t1 = llama_time_us();
    prefill_ms = (double)(t1 - t0) / 1000.0;

    n_embd = llama_model_n_embd(model);
    embd = llama_get_embeddings_ith(ctx, n_in - 1);
    if (embd == NULL) {
        embd = llama_get_embeddings(ctx);
    }
    if (embd != NULL) {
        for (int32_t i = 0; i < n_embd; ++i) {
            norm_sq += (double)embd[i] * (double)embd[i];
        }
    }

    printf("{\n");
    printf("  \"mode\": \"embed\",\n");
    printf("  \"model\": "); print_json_escaped(model_path); printf(",\n");
    printf("  \"n_input_tokens\": %d,\n", n_in);
    printf("  \"n_embd\": %d,\n", n_embd);
    printf("  \"l2_norm\": %.6f,\n", sqrt(norm_sq));
    printf("  \"tokenize_ms\": %.3f,\n", tokenize_ms);
    printf("  \"prefill_ms\": %.3f,\n", prefill_ms);
    printf("  \"rss_mib\": %.2f\n", get_rss_mib());
    printf("}\n");

    free(tokens);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *kv_test_path = NULL;
    const char *prompt = "Explain why cross-model KV-cache sharing requires architectural compatibility.";
    const char *mode = "bench";
    int32_t max_tokens = 32;
    uint32_t n_ctx = 512u;
    int32_t n_threads = 4;
    int rc = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--kv-test") == 0 && i + 1 < argc) {
            kv_test_path = argv[++i];
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            mode = argv[++i];
        } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--n-ctx") == 0 && i + 1 < argc) {
            n_ctx = (uint32_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        }
    }

    if (model_path == NULL) {
        fprintf(stderr, "Usage: %s --model <path.gguf> [--mode bench|kv-compat|embed] [--kv-test <other.gguf>] [--prompt <text>] [--max-tokens N]\n", argv[0]);
        return 1;
    }

    llama_log_set(quiet_log_cb, NULL);
    llama_backend_init();

    if (strcmp(mode, "bench") == 0) {
        rc = run_bench(model_path, prompt, max_tokens, n_ctx, n_threads);
    } else if (strcmp(mode, "kv-compat") == 0) {
        rc = run_kv_compat(model_path, kv_test_path, prompt, n_ctx, n_threads);
    } else if (strcmp(mode, "embed") == 0) {
        rc = run_embed(model_path, prompt, n_ctx, n_threads);
    } else {
        fprintf(stderr, "Unknown mode: %s\n", mode);
        rc = 1;
    }

    llama_backend_free();
    return rc;
}
