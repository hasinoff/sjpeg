// Copyright 2017 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// AVX2/SIMD implementation of the inner loop of SjpegRiskiness() (jpeg_tools.cc).
//
// Method description:
// SjpegRiskiness() scores visual sharpness and complexity of neighboring pixels
// across two consecutive rows (row1 = above, row2 = below) using a precomputed
// 3-way sharpness table (kSharpnessScore, size 343 * 343).
//
// Upstream originally evaluated an AVX2 variant using vpgatherdd, which was
// disabled by default (SJPEG_USE_AVX2_RISKINESS) due to severe gather stalls on
// older hardware (Haswell/Excavator) and high port-5 pressure.
//
// This implementation eliminates hardware gather instructions entirely:
//  1. Precomputes an array of row pointers (kRowTable[343], 2.7 KB), which fits
//     entirely within L1D cache (typically 32 KB or 48 KB).
//  2. Evaluates the 3-way pair lookups using direct L1D memory loads on the
//     CPU's dual load ports, avoiding any gather latency or microcode overhead.
//  3. Packs 8 scores into a 128-bit vector as 16-bit integers.
//  4. Uses 16-bit SIMD math to evaluate the chroma neutrality test, noise
//     threshold test, and score accumulation (using _mm_madd_epi16).
//  5. Unrolls by 2 (16 pixels per iteration in the main loop) to maximize
//     instruction throughput and pipeline latency hiding.
//
// Note on instruction set: All vector operations in Process8Pixels use 128-bit
// SSE2 instructions. When compiled with -mavx2, the compiler emits 3-operand
// VEX-prefixed instructions (e.g. vpaddd, vpsubw) to eliminate register copies.
//
// Author: Skal (pascal.massimino@gmail.com)
//         Sam Hasinoff (hasinoff@google.com)

#define SJPEG_NEED_ASM_HEADERS
#include "sjpegi.h"

#if defined(SJPEG_USE_AVX2)

namespace sjpeg {

constexpr int kRGB3 = 343;  // kRGBSize * kRGBSize * kRGBSize

alignas(64) static const uint8_t* kRowTable[kRGB3];

static void InitRowTable() {
  for (int i = 0; i < kRGB3; ++i) {
    kRowTable[i] = &kSharpnessScore[i * kRGB3];
  }
}

static struct RowTableInitializer {
  RowTableInitializer() { InitRowTable(); }
} kInitRowTable;

// Horizontal sum helpers for 128-bit integer vectors.
static inline int32_t HorizontalSumEpi32(__m128i v) {
  const __m128i hi = _mm_unpackhi_epi64(v, v);
  const __m128i sum = _mm_add_epi32(v, hi);
  const __m128i hi2 = _mm_shuffle_epi32(sum, _MM_SHUFFLE(1, 1, 1, 1));
  return _mm_cvtsi128_si32(_mm_add_epi32(sum, hi2));
}

static inline int32_t HorizontalSumEpi16(__m128i v) {
  const __m128i ones = _mm_set1_epi16(1);
  return HorizontalSumEpi32(_mm_madd_epi16(v, ones));
}

// Evaluates 8 pixels of riskiness scores starting at (row1, row2).
// Computes 3-way pair sharpness scores via 2D row table lookups and vectorizes
// chroma neutrality tests, noise thresholding, and score accumulation.
__attribute__((always_inline))
static inline void Process8Pixels(
    const uint16_t* const row1, const uint16_t* const row2,
    __m128i min_16, __m128i max_16, __m128i noise_vec_16, __m128i ones_16,
    __m128i* const gray_vec_16, __m128i* const num_vec_16,
    __m128i* const sum_vec_32) {
  // 1. Neutral chroma (gray level) test on row1 samples:
  //    gray_min <= idx0 < gray_min + s
  const __m128i r1_0 =
      _mm_loadu_si128(reinterpret_cast<const __m128i*>(row1));
  const __m128i ge_mask = _mm_cmpgt_epi16(r1_0, min_16);
  const __m128i lt_mask = _mm_cmpgt_epi16(max_16, r1_0);
  *gray_vec_16 = _mm_sub_epi16(*gray_vec_16, _mm_and_si128(ge_mask, lt_mask));

  // 2. 3-way sharpness score lookups via kRowTable:
  //    score = table[idx0 + K*idx1] + table[idx0 + K*idx2] + table[idx1 + K*idx2]
  //    where idx0 = row1[k], idx1 = row1[k + 1], idx2 = row2[k].
  auto score_at = [&](int k) -> uint32_t {
    const uint8_t* const row_v = kRowTable[row2[k]];
    const uint8_t* const row_u = kRowTable[row1[k + 1]];
    return (uint32_t)(row_u[row1[k]] + row_v[row1[k]] + row_v[row1[k + 1]]);
  };

  // 3. Pack 8 scores into a 128-bit SIMD register as uint16_t elements.
  const uint64_t w0 = (uint64_t)(score_at(0) | (score_at(1) << 16)) |
                      ((uint64_t)(score_at(2) | (score_at(3) << 16)) << 32);
  const uint64_t w1 = (uint64_t)(score_at(4) | (score_at(5) << 16)) |
                      ((uint64_t)(score_at(6) | (score_at(7) << 16)) << 32);
  const __m128i scores_16 = _mm_set_epi64x(w1, w0);

  // 4. Vectorized noise threshold filter and accumulation.
  const __m128i score_mask = _mm_cmpgt_epi16(scores_16, noise_vec_16);
  *num_vec_16 = _mm_sub_epi16(*num_vec_16, score_mask);
  *sum_vec_32 = _mm_add_epi32(*sum_vec_32,
      _mm_madd_epi16(_mm_and_si128(scores_16, score_mask), ones_16));
}

// Processes 'size' samples in [0, size), 8 at a time.
// Returns how many samples were actually consumed (a multiple of 8, <= size).
// Caller needs to handle the [return value, size) remainder using C-version.
int RiskinessScoreRowAVX2(const uint16_t* row1, const uint16_t* row2,
                          int size, int noise_level,
                          int64_t* const score_sum, int64_t* const score_num,
                          int64_t* const gray_num) {
  if (kRowTable[0] == nullptr) {
    InitRowTable();
  }

  const int s = kRGBSize;
  const int gray = (s / 2) * (1 + s) * s;   // gray level for y=0,u=128,v=128
  const int gray_min = gray - gray % s;

  const __m128i min_16 = _mm_set1_epi16(gray_min - 1);
  const __m128i max_16 = _mm_set1_epi16(gray_min + s);
  const __m128i noise_vec_16 = _mm_set1_epi16(noise_level);
  const __m128i ones_16 = _mm_set1_epi16(1);

  // 8-lane accumulators, reduced post-loop. Overflow-safe since kMaxDimension
  // is 65535 => even the worst case (max score on every iteration):
  // num_vec_16 and gray_vec_16 accumulate at most 65535/8 = 8192 counts per
  // lane per row, which stays safely within the signed 16-bit limit (32767).
  // sum_vec_32 uses 32-bit accumulators via _mm_madd_epi16, which stays well
  // under the 32-bit limit (65534/8 * 765 ~= 6.3M).
  __m128i sum_vec_32 = _mm_setzero_si128();  // scores above the noise level
  __m128i num_vec_16 = _mm_setzero_si128();  // number of sum_vec
  __m128i gray_vec_16 = _mm_setzero_si128(); // samples with neutral chroma

  int i = 0;
  // Main unrolled loop: process 32 pixels per iteration across 4 pipelined blocks.
  for (; i + 32 <= size; i += 32) {
    Process8Pixels(row1 + i + 0, row2 + i + 0, min_16, max_16, noise_vec_16,
                   ones_16, &gray_vec_16, &num_vec_16, &sum_vec_32);
    Process8Pixels(row1 + i + 8, row2 + i + 8, min_16, max_16, noise_vec_16,
                   ones_16, &gray_vec_16, &num_vec_16, &sum_vec_32);
    Process8Pixels(row1 + i + 16, row2 + i + 16, min_16, max_16, noise_vec_16,
                   ones_16, &gray_vec_16, &num_vec_16, &sum_vec_32);
    Process8Pixels(row1 + i + 24, row2 + i + 24, min_16, max_16, noise_vec_16,
                   ones_16, &gray_vec_16, &num_vec_16, &sum_vec_32);
  }

  // Trailing 16-pixel block.
  for (; i + 16 <= size; i += 16) {
    Process8Pixels(row1 + i, row2 + i, min_16, max_16, noise_vec_16, ones_16,
                   &gray_vec_16, &num_vec_16, &sum_vec_32);
    Process8Pixels(row1 + i + 8, row2 + i + 8, min_16, max_16, noise_vec_16,
                   ones_16, &gray_vec_16, &num_vec_16, &sum_vec_32);
  }

  // Trailing 8-pixel block if size is not a multiple of 16.
  for (; i + 8 <= size; i += 8) {
    Process8Pixels(row1 + i, row2 + i, min_16, max_16, noise_vec_16, ones_16,
                   &gray_vec_16, &num_vec_16, &sum_vec_32);
  }

  *score_sum += HorizontalSumEpi32(sum_vec_32);
  *score_num += HorizontalSumEpi16(num_vec_16);
  *gray_num += HorizontalSumEpi16(gray_vec_16);
  return i;
}

}  // namespace sjpeg

#endif  // SJPEG_USE_AVX2
