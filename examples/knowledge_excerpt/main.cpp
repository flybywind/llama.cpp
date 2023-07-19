// Defines sigaction on msys:
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "build-info.h"
#include "common.h"
#include "llama.h"

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <signal.h>
#include <windows.h>
#endif

#if defined(_MSC_VER)
#pragma warning(disable : 4244 4267)  // possible loss of data
#endif

static console_state con_st;
static llama_context** g_ctx;

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__)) || defined(_WIN32)
void sigint_handler(int signo) {
    if (signo == SIGINT) {
        console_cleanup(con_st);
        printf("\n");
        llama_print_timings(*g_ctx);
        _exit(130);
    }
}
#endif

int main(int argc, char** argv) {
    gpt_params params;

    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    // save choice to use color for later
    // (note for later: this is a slightly awkward choice)
    con_st.use_color = params.use_color;
    con_st.multiline_input = params.multiline_input;
    console_init(con_st);
    atexit([]() { console_cleanup(con_st); });

    if (params.perplexity) {
        printf("\n************\n");
        printf("%s: please use the 'perplexity' tool for perplexity calculations\n", __func__);
        printf("************\n\n");

        return 0;
    }
    if (params.interactive) {
        printf("\n************\n");
        printf("%s: don't support interactive mode yet\n", __func__);
        printf("************\n\n");

        return 0;
    }
    if (params.embedding) {
        printf("\n************\n");
        printf("%s: please use the 'embedding' tool for embedding calculations\n", __func__);
        printf("************\n\n");

        return 0;
    }

    if (params.rope_freq_base != 10000.0) {
        fprintf(stderr, "%s: warning: changing RoPE frequency base to %g (default 10000.0)\n", __func__, params.rope_freq_base);
    }

    if (params.rope_freq_scale != 1.0) {
        fprintf(stderr, "%s: warning: scaling RoPE frequency by %g (default 1.0)\n", __func__, params.rope_freq_scale);
    }

    if (params.n_ctx > 2048) {
        fprintf(stderr,
                "%s: warning: base model only supports context sizes no greater than 2048 tokens (%d specified);"
                " you are on your own\n",
                __func__, params.n_ctx);
    } else if (params.n_ctx < 8) {
        fprintf(stderr, "%s: warning: minimum context size is 8, using minimum size.\n", __func__);
        params.n_ctx = 8;
    }

    fprintf(stderr, "%s: build = %d (%s)\n", __func__, BUILD_NUMBER, BUILD_COMMIT);

    if (params.seed == LLAMA_DEFAULT_SEED) {
        params.seed = time(NULL);
    }

    fprintf(stderr, "%s: seed  = %u\n", __func__, params.seed);

    std::mt19937 rng(params.seed);
    if (params.random_prompt) {
        params.prompt = gpt_random_prompt(rng);
    }

    llama_backend_init(params.numa);

    llama_model* model;
    llama_context* ctx;
    llama_context* ctx_guidance = NULL;
    g_ctx = &ctx;

    // load the model and apply lora adapter, if any
    std::tie(model, ctx) = llama_init_from_gpt_params(params);
    if (model == NULL) {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return 1;
    }

    // print system information
    {
        fprintf(stderr, "\n");
        fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
                params.n_threads, std::thread::hardware_concurrency(), llama_print_system_info());
    }

    // determine the maximum memory usage needed to do inference for the given n_batch and n_predict parameters
    // uncomment the "used_mem" line in llama.cpp to see the results
    if (params.mem_test) {
        {
            const std::vector<llama_token> tmp(params.n_batch, llama_token_bos());
            llama_eval(ctx, tmp.data(), tmp.size(), 0, params.n_threads);
        }

        {
            const std::vector<llama_token> tmp = {
                0,
            };
            llama_eval(ctx, tmp.data(), tmp.size(), params.n_predict - 1, params.n_threads);
        }

        llama_print_timings(ctx);
        llama_free(ctx);
        llama_free_model(model);

        return 0;
    }

    // tokenize the prompt
    std::vector<llama_token> embd_inp;

    // Add a space in front of the first character to match OG llama tokenizer behavior
    params.prompt.insert(0, 1, ' ');

    embd_inp = ::llama_tokenize(ctx, params.prompt, true);

    // Tokenize knowledge content
    std::vector<llama_token> guidance_inp;
    int original_prompt_len = 0;
    if (params.knowledge_scale > 0.f) {
        struct llama_context_params lparams = llama_context_params_from_gpt_params(params);
        ctx_guidance = llama_new_context_with_model(model, lparams);
        params.knowledge_str.insert(0, 1, ' ');
        guidance_inp = ::llama_tokenize(ctx_guidance, params.knowledge_str, true);
    }

    const int n_ctx = llama_n_ctx(ctx);

    if ((int)embd_inp.size() > n_ctx - 4) {
        fprintf(stderr, "%s: error: prompt is too long (%d tokens, max %d)\n", __func__, (int)embd_inp.size(), n_ctx - 4);
        return 1;
    }

    // determine newline token
    auto llama_token_newline = ::llama_tokenize(ctx, "\n", false);

    if (params.verbose_prompt) {
        fprintf(stderr, "\n");
        fprintf(stderr, "%s: prompt: '%s'\n", __func__, params.prompt.c_str());
        fprintf(stderr, "%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
        for (int i = 0; i < (int)embd_inp.size(); i++) {
            fprintf(stderr, "%6d -> '%s'\n", embd_inp[i], llama_token_to_str(ctx, embd_inp[i]));
        }

        if (ctx_guidance) {
            fprintf(stderr, "\n");
            fprintf(stderr, "%s: negative prompt: '%s'\n", __func__, params.cfg_negative_prompt.c_str());
            fprintf(stderr, "%s: number of tokens in negative prompt = %zu\n", __func__, guidance_inp.size());
            for (int i = 0; i < (int)guidance_inp.size(); i++) {
                fprintf(stderr, "%6d -> '%s'\n", guidance_inp[i], llama_token_to_str(ctx, guidance_inp[i]));
            }
        }

        if (params.n_keep > 0) {
            fprintf(stderr, "%s: static prompt based on n_keep: '", __func__);
            for (int i = 0; i < params.n_keep; i++) {
                fprintf(stderr, "%s", llama_token_to_str(ctx, embd_inp[i]));
            }
            fprintf(stderr, "'\n");
        }
        fprintf(stderr, "\n");
    }

    fprintf(stderr, "sampling: repeat_last_n = %d, repeat_penalty = %f, presence_penalty = %f, frequency_penalty = %f, top_k = %d, tfs_z = %f, top_p = %f, typical_p = %f, temp = %f, mirostat = %d, mirostat_lr = %f, mirostat_ent = %f\n",
            params.repeat_last_n, params.repeat_penalty, params.presence_penalty, params.frequency_penalty, params.top_k, params.tfs_z, params.top_p, params.typical_p, params.temp, params.mirostat, params.mirostat_eta, params.mirostat_tau);
    fprintf(stderr, "generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n", n_ctx, params.n_batch, params.n_predict, params.n_keep);
    fprintf(stderr, "\n\n");

    // TODO: replace with ring-buffer
    std::vector<llama_token> last_n_tokens(n_ctx);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

    bool input_echo = true;

    int n_past = 0;
    int n_remain = params.n_predict;
    int n_consumed = 0;
    int n_past_guidance = 0;

    // the first thing we will do is to output the prompt, so set color accordingly
    console_set_color(con_st, CONSOLE_COLOR_PROMPT);

    std::vector<llama_token> embd;
    std::vector<llama_token> embd_guidance;

    // out of user input, sample next token
    const float temp = params.temp;
    const int32_t top_k = params.top_k <= 0 ? llama_n_vocab(ctx) : params.top_k;
    const float top_p = params.top_p;
    const float tfs_z = params.tfs_z;
    const float typical_p = params.typical_p;
    const int32_t repeat_last_n = params.repeat_last_n < 0 ? n_ctx : params.repeat_last_n;
    const float repeat_penalty = params.repeat_penalty;
    const float alpha_presence = params.presence_penalty;
    const float alpha_frequency = params.frequency_penalty;
    const int mirostat = params.mirostat;
    const float mirostat_tau = params.mirostat_tau;
    const float mirostat_eta = params.mirostat_eta;
    const bool penalize_nl = params.penalize_nl;

    llama_token id = 0;

    {
        auto logits = llama_get_logits(ctx);
        auto n_vocab = llama_n_vocab(ctx);

        // Apply params.logit_bias map
        // for (auto it = params.logit_bias.begin(); it != params.logit_bias.end(); it++) {
        //     logits[it->first] += it->second;
        // }

        std::vector<llama_token_data> candidates;
        candidates.reserve(n_vocab);
        for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
            candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
        }

        llama_token_data_array candidates_p = {candidates.data(), candidates.size(), false};

        // Apply penalties
        float nl_logit = logits[llama_token_nl()];
        auto last_n_repeat = std::min(std::min((int)last_n_tokens.size(), repeat_last_n), n_ctx);
        llama_sample_repetition_penalty(ctx, &candidates_p,
                                        last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                                        last_n_repeat, repeat_penalty);
        llama_sample_frequency_and_presence_penalties(ctx, &candidates_p,
                                                      last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                                                      last_n_repeat, alpha_frequency, alpha_presence);
        if (ctx_guidance) {
            if (guidance_inp.size() > 0) {
                if (llama_eval(ctx_guidance, guidance_inp.data(), guidance_inp.size(), std::min<int>(guidance_inp.size(), n_ctx), params.n_threads)) {
                    fprintf(stderr, "%s : failed to eval the guidance\n", __func__);
                    return 1;
                }
            }

            const int n_embd = llama_n_embd(ctx_guidance);
            const auto guidance_emb = llama_get_embeddings(ctx_guidance);
        }
        if (!penalize_nl) {
            logits[llama_token_nl()] = nl_logit;
        }

        if (temp <= 0) {
            // Greedy sampling
            id = llama_sample_token_greedy(ctx, &candidates_p);
        } else {
            if (mirostat == 1) {
                static float mirostat_mu = 2.0f * mirostat_tau;
                const int mirostat_m = 100;
                llama_sample_temperature(ctx, &candidates_p, temp);
                id = llama_sample_token_mirostat(ctx, &candidates_p, mirostat_tau, mirostat_eta, mirostat_m, &mirostat_mu);
            } else if (mirostat == 2) {
                static float mirostat_mu = 2.0f * mirostat_tau;
                llama_sample_temperature(ctx, &candidates_p, temp);
                id = llama_sample_token_mirostat_v2(ctx, &candidates_p, mirostat_tau, mirostat_eta, &mirostat_mu);
            } else {
                // Temperature sampling
                llama_sample_top_k(ctx, &candidates_p, top_k, 1);
                llama_sample_tail_free(ctx, &candidates_p, tfs_z, 1);
                llama_sample_typical(ctx, &candidates_p, typical_p, 1);
                llama_sample_top_p(ctx, &candidates_p, top_p, 1);
                llama_sample_temperature(ctx, &candidates_p, temp);
                id = llama_sample_token(ctx, &candidates_p);
            }
        }
        // printf("`%d`", candidates_p.size);

        last_n_tokens.erase(last_n_tokens.begin());
        last_n_tokens.push_back(id);
    }

    // replace end of text token with newline token when in interactive mode
    if (id == llama_token_eos() && params.interactive && !params.instruct) {
        id = llama_token_newline.front();
        if (params.antiprompt.size() != 0) {
            // tokenize and inject first reverse prompt
            const auto first_antiprompt = ::llama_tokenize(ctx, params.antiprompt.front(), false);
            embd_inp.insert(embd_inp.end(), first_antiprompt.begin(), first_antiprompt.end());
        }
    }

    // add it to the context
    embd.push_back(id);

    // echo this to console
    input_echo = true;

    // decrement remaining sampling budget
    --n_remain;
    // display text
    if (input_echo) {
        for (auto id : embd) {
            printf("%s", llama_token_to_str(ctx, id));
        }
        fflush(stdout);
    }
    // reset color to default if we there is no pending user input
    if (input_echo && (int)embd_inp.size() == n_consumed) {
        console_set_color(con_st, CONSOLE_COLOR_DEFAULT);
    }

    // if not currently processing queued inputs;
    if ((int)embd_inp.size() <= n_consumed) {
        // check for reverse prompt
        if (params.antiprompt.size()) {
            std::string last_output;
            for (auto id : last_n_tokens) {
                last_output += llama_token_to_str(ctx, id);
            }

            is_antiprompt = false;
            // Check if each of the reverse prompts appears at the end of the output.
            // If we're not running interactively, the reverse prompt might be tokenized with some following characters
            // so we'll compensate for that by widening the search window a bit.
            for (std::string& antiprompt : params.antiprompt) {
                size_t extra_padding = params.interactive ? 0 : 2;
                size_t search_start_pos = last_output.length() > static_cast<size_t>(antiprompt.length() + extra_padding)
                                              ? last_output.length() - static_cast<size_t>(antiprompt.length() + extra_padding)
                                              : 0;

                if (last_output.find(antiprompt.c_str(), search_start_pos) != std::string::npos) {
                    if (params.interactive) {
                        is_interacting = true;
                        console_set_color(con_st, CONSOLE_COLOR_USER_INPUT);
                    }
                    is_antiprompt = true;
                    fflush(stdout);
                    break;
                }
            }
        }

        if (n_past > 0) {
            is_interacting = false;
        }
    }

    // end of text token
    if (!embd.empty() && embd.back() == llama_token_eos()) {
        if (params.instruct) {
            is_interacting = true;
        } else {
            fprintf(stderr, " [end of text]\n");
            break;
        }
    }

    llama_print_timings(ctx);
    if (ctx_guidance) {
        llama_free(ctx_guidance);
    }
    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

    return 0;
}
