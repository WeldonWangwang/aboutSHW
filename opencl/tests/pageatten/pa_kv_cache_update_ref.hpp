
/*
 * Copyright (c) 2020-2025, Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#include <cm/cm.h>
#include <cm/cmtl.h>
#include <cmath>
#include <limits>

#ifndef KEY_BLOCK_DATA_BYTES
#define KEY_BLOCK_DATA_BYTES (PAGED_ATTENTION_BLOCK_SIZE * K_HEAD_SIZE)
#endif
#ifndef KEY_BLOCK_SCALE_BYTES
#define KEY_BLOCK_SCALE_BYTES (PAGED_ATTENTION_BLOCK_SIZE * sizeof(half))
#endif
#ifndef KEY_BLOCK_ZP_BYTES
#define KEY_BLOCK_ZP_BYTES (PAGED_ATTENTION_BLOCK_SIZE * sizeof(half))
#endif
#ifndef KEY_BLOCK_STRIDE_BYTES
#define KEY_BLOCK_STRIDE_BYTES (KEY_BLOCK_DATA_BYTES + KEY_BLOCK_SCALE_BYTES + KEY_BLOCK_ZP_BYTES)
#endif

#ifndef VALUE_BLOCK_DATA_BYTES
#define VALUE_BLOCK_DATA_BYTES (PAGED_ATTENTION_BLOCK_SIZE * V_HEAD_SIZE)
#endif
#ifndef VALUE_BLOCK_SCALE_BYTES
#define VALUE_BLOCK_SCALE_BYTES (PAGED_ATTENTION_BLOCK_SIZE * sizeof(half))
#endif
#ifndef VALUE_BLOCK_ZP_BYTES
#define VALUE_BLOCK_ZP_BYTES (PAGED_ATTENTION_BLOCK_SIZE * sizeof(half))
#endif
#ifndef VALUE_BLOCK_STRIDE_BYTES
#define VALUE_BLOCK_STRIDE_BYTES (VALUE_BLOCK_DATA_BYTES + VALUE_BLOCK_SCALE_BYTES + VALUE_BLOCK_ZP_BYTES)
#endif

#ifndef ATTR
#define ATTR [[type("svmptr_t")]]
#define ATTR_BUF [[type("buffer_t")]]
#endif

constexpr uint wg_size = WG_SIZE;

extern "C" _GENX_MAIN_ void pa_kv_cache_update(
    const half* key [[type("svmptr_t")]],
    const half* value [[type("svmptr_t")]],
    const int32_t* past_lens [[type("svmptr_t")]],
    const int32_t* block_indices [[type("svmptr_t")]],
    const int32_t* block_indices_begins [[type("svmptr_t")]],
    const int32_t* subsequence_begins [[type("svmptr_t")]],
#if KV_CACHE_COMPRESSION_PER_TOKEN
    uint8_t* key_cache [[type("svmptr_t")]],
    uint8_t* value_cache [[type("svmptr_t")]],
#else
    half* key_cache [[type("svmptr_t")]],
    half* value_cache [[type("svmptr_t")]],
#endif
    uint32_t key_pitch,
    uint32_t value_pitch,
    uint32_t batch_size_in_sequences) {
    // # key:   [batch_size_in_tokens, num_kv_heads * k_head_size]
    // # value  [batch_size_in_tokens, num_kv_heads * v_head_size]
    // # key_cache:   [num_blocks, num_heads, block_size, k_head_size]
    // # value_cache: [num_blocks, num_heads, block_size, v_head_size]
    // 
    // # past_lens: [sequences_num]
    // # subsequence_begins: [sequences_num + 1]
    // # block_indices: [used_blocks_num]
    // # block_indices_begins: [sequences_num + 1]

    // wg_count = aligned_to(batch_size_in_tokens, wg_size) // wg_size
    // # GWS [1, num_heads, wg_count * wg_size]
    // # LWS [1, 1, wg_size]

    const auto head_idx = cm_group_id(1);
    const auto wg_id = cm_group_id(2);
    //const auto wg_local_id = cm_local_id(2);
    //const auto local_size = cm_local_size(2);

    // static_assert(local_size == wg_size);

    // const uint token_idx = wg_id * local_size + wg_local_id;
    const uint token_idx = cm_global_id(2);

    // token_idx -> subsequence_idx
    if (token_idx >= subsequence_begins[batch_size_in_sequences]) return;
    uint subsequence_idx = 0;
    for (uint i = 0; i < batch_size_in_sequences; i++) {
        if (token_idx >= subsequence_begins[i] && token_idx < subsequence_begins[i + 1]) {
            subsequence_idx = i;
            break;
        }
    }
    // printf("wg:%d.%d, token_idx: %d, subsequence_idx: %d\n", wg_id, wg_local_id, token_idx, subsequence_idx);

    const uint subsequence_begin_idx = subsequence_begins[subsequence_idx];

    const uint past_len = past_lens[subsequence_idx];

    const uint current_block_idx = (past_len + token_idx - subsequence_begin_idx) / PAGED_ATTENTION_BLOCK_SIZE;
    const uint token_start_pos = (past_len + token_idx - subsequence_begin_idx) % PAGED_ATTENTION_BLOCK_SIZE;

    const uint block_offset = block_indices_begins[subsequence_idx] + current_block_idx;

    #if KV_CACHE_COMPRESSION_PER_TOKEN
    constexpr uint KEY_DATA_BYTES = KEY_BLOCK_DATA_BYTES;
    constexpr uint KEY_SCALE_BYTES = KEY_BLOCK_SCALE_BYTES;
    constexpr uint KEY_ZP_BYTES = KEY_BLOCK_ZP_BYTES;
    constexpr uint KEY_STRIDE_BYTES = KEY_BLOCK_STRIDE_BYTES;
    constexpr uint VALUE_DATA_BYTES = VALUE_BLOCK_DATA_BYTES;
    constexpr uint VALUE_SCALE_BYTES = VALUE_BLOCK_SCALE_BYTES;
    constexpr uint VALUE_ZP_BYTES = VALUE_BLOCK_ZP_BYTES;
    constexpr uint VALUE_STRIDE_BYTES = VALUE_BLOCK_STRIDE_BYTES;

    auto clamp_to_byte = [](float value) -> uchar {
        if (value < 0.0f)
            return 0;
        if (value > 255.0f)
            return 255;
        return static_cast<uchar>(std::nearbyint(value));
    };

    auto quantize_and_store_token = [&](vector<half, K_HEAD_SIZE> data, uchar* out, uint block_offset, uint token_pos) {
            uint token_offset = block_offset + token_pos * K_HEAD_SIZE;
            uint scale_offset = block_offset + KEY_DATA_BYTES + token_pos * sizeof(half);
            uint zp_offset = block_offset + KEY_DATA_BYTES + KEY_SCALE_BYTES + token_pos * sizeof(half);

            half max_val = cm_reduced_max<half>(data);
            half min_val = cm_reduced_min<half>(data);
            half scale_val = half(0.0);
            half zp_val = half(0.0);
            if(max_val == min_val) {
                scale_val = half(0.0);
                zp_val = max_val;
            } else {
                scale_val = 255.0 / (max_val - min_val);
                zp_val = (0.0 - min_val) * scale_val;
            }
            vector<half, K_HEAD_SIZE> quant_data = cm_mul<half>(data, scale_val) + zp_val;
            vector<uchar, K_HEAD_SIZE> data_u8 = cm_rnde<uchar, K_HEAD_SIZE>(quant_data);
            cm_ptr_store<uint32_t, K_HEAD_SIZE / 4>((uint32_t*)(out + token_offset), 0, data_u8.format<uint32_t>());
            half* out_scale = reinterpret_cast<half*>(out + scale_offset);
            half* out_zp = reinterpret_cast<half*>(out + zp_offset);
            out_scale[0] = (max_val - min_val) / 255.0;
            out_zp[0] = zp_val;
    };

    auto quantize_and_store_channel = [&](vector<half, K_HEAD_SIZE> data, uchar* out, uint block_offset, uint token_pos, uint block_idx, uint past_tokens) {
            constexpr uint block_size = PAGED_ATTENTION_BLOCK_SIZE;
            uint block_tokens_before = 0;
            uint full_blocks = past_tokens / block_size;
            uint remainder = past_tokens % block_size;
            if (block_idx < full_blocks) {
                block_tokens_before = block_size;
            } else if (block_idx == full_blocks) {
                block_tokens_before = remainder;
            }

            uint block_tokens_after = block_tokens_before;
            if (token_pos + 1 > block_tokens_after) block_tokens_after = token_pos + 1;
            if (block_tokens_after > block_size) block_tokens_after = block_size;

            uchar* data_base = out + block_offset;
            half* scale_base = reinterpret_cast<half*>(out + block_offset + KEY_DATA_BYTES);
            half* zp_base = reinterpret_cast<half*>(out + block_offset + KEY_DATA_BYTES + KEY_SCALE_BYTES);

            for (uint channel = 0; channel < K_HEAD_SIZE; ++channel) {
                float values[block_size];
                half scale_h = scale_base[channel];
                half zp_h = zp_base[channel];
                float scale = static_cast<float>(scale_h);
                float zp = static_cast<float>(zp_h);
                for (uint t = 0; t < block_tokens_after; ++t) {
                    if (t == token_pos) {
                        values[t] = static_cast<float>(data.select<1, 1>(channel)[0]);
                    } else {
                        uchar q = data_base[t * K_HEAD_SIZE + channel];
                        if (scale == 0.0f) {
                            values[t] = zp;
                        } else {
                            values[t] = (static_cast<float>(q) - zp) * scale;
                        }
                    }
                }

                float min_value = std::numeric_limits<float>::max();
                float max_value = std::numeric_limits<float>::lowest();
                for (uint t = 0; t < block_tokens_after; ++t) {
                    float v = values[t];
                    if (v < min_value) min_value = v;
                    if (v > max_value) max_value = v;
                }

                float scale_val = 0.0f;
                float zp_val = 0.0f;
                if (max_value != min_value) {
                    float diff_value = max_value - min_value;
                    scale_val = 255.0f / diff_value;
                    zp_val = -min_value * scale_val;
                } else {
                    zp_val = min_value;
                }

                for (uint t = 0; t < block_tokens_after; ++t) {
                    float q = values[t] * scale_val + zp_val;
                    data_base[t * K_HEAD_SIZE + channel] = clamp_to_byte(q);
                }

                for (uint t = block_tokens_after; t < block_size; ++t) {
                    data_base[t * K_HEAD_SIZE + channel] = 0;
                }

                scale_base[channel] = static_cast<half>((max_value - min_value) / 255.0f);
                zp_base[channel] = static_cast<half>(zp_val);
            }
    };

    auto quantize_and_store_value_token = [&](vector<half, V_HEAD_SIZE> data, uchar* out, uint block_offset, uint token_pos) {
            uint token_offset = block_offset + token_pos * V_HEAD_SIZE;
            uint scale_offset = block_offset + VALUE_DATA_BYTES + token_pos * sizeof(half);
            uint zp_offset = block_offset + VALUE_DATA_BYTES + VALUE_SCALE_BYTES + token_pos * sizeof(half);

            half max_val = cm_reduced_max<half>(data);
            half min_val = cm_reduced_min<half>(data);
            half scale_val = half(0.0);
            half zp_val = half(0.0);
            if(max_val == min_val) {
                scale_val = half(0.0);
                zp_val = max_val;
            } else {
                scale_val = 255.0 / (max_val - min_val);
                zp_val = (0.0 - min_val) * scale_val;
            }
            vector<half, V_HEAD_SIZE> quant_data = cm_mul<half>(data, scale_val) + zp_val;
            vector<uchar, V_HEAD_SIZE> data_u8 = cm_rnde<uchar, V_HEAD_SIZE>(quant_data);
            cm_ptr_store<uint32_t, V_HEAD_SIZE / 4>((uint32_t*)(out + token_offset), 0, data_u8.format<uint32_t>());
            half* out_scale = reinterpret_cast<half*>(out + scale_offset);
            half* out_zp = reinterpret_cast<half*>(out + zp_offset);
            out_scale[0] = (max_val - min_val) / 255.0;
            out_zp[0] = zp_val;
    };
    #endif
    {
        uint block_identifier = block_indices[block_offset];
        #if KV_CACHE_COMPRESSION_PER_TOKEN
        uint block_k_base_offset = (block_identifier * KV_HEADS_NUM + head_idx) * KEY_STRIDE_BYTES;
        #else
        uint block_k_base_offset = (block_identifier * KV_HEADS_NUM + head_idx) * ADJUSTED_K_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE;
        #endif
        uint key_out_offset = block_k_base_offset + token_start_pos * K_HEAD_SIZE;
        uint key_in_offset = token_idx * key_pitch + head_idx * K_HEAD_SIZE;
        vector<half, K_HEAD_SIZE> key_data;
        key_data.format<int>() = cm_ptr_load<int, K_HEAD_SIZE / 2>((int*)key, key_in_offset * (int)sizeof(half));

        #if KV_CACHE_COMPRESSION_PER_TOKEN
            #if KV_CACHE_COMPRESSION_PER_TOKEN == 1
                quantize_and_store_token(key_data, (uchar*)key_cache, block_k_base_offset, token_start_pos);
            #else
                quantize_and_store_channel(key_data, (uchar*)key_cache, block_k_base_offset, token_start_pos, current_block_idx, past_len);
            #endif
        #else
            cm_ptr_store<int, K_HEAD_SIZE / 2>((int*)key_cache, key_out_offset * (int)sizeof(half), key_data.format<int>());
        #endif
    }
    {
        uint block_identifier = block_indices[block_offset];
        #if KV_CACHE_COMPRESSION_PER_TOKEN
        uint block_v_base_offset = (block_identifier * KV_HEADS_NUM + head_idx) * VALUE_STRIDE_BYTES;
        #else
        uint block_v_base_offset = (block_identifier * KV_HEADS_NUM + head_idx) * ADJUSTED_V_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE;
        #endif
        uint value_out_offset = block_v_base_offset + token_start_pos * V_HEAD_SIZE;
        uint value_in_offset = token_idx * value_pitch + head_idx * V_HEAD_SIZE;
        vector<half, V_HEAD_SIZE> value_data;
        value_data.format<int>() = cm_ptr_load<int, V_HEAD_SIZE / 2>((int*)value, value_in_offset * (int)sizeof(half));

        #if KV_CACHE_COMPRESSION_PER_TOKEN
            quantize_and_store_value_token(value_data, (uchar*)value_cache, block_v_base_offset, token_start_pos);
        #else
            cm_ptr_store<int, V_HEAD_SIZE / 2>((int*)value_cache, value_out_offset * (int)sizeof(half), value_data.format<int>());
        #endif
    }
}