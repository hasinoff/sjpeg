// Copyright 2026 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//  Multi-threaded scan engines using JPEG restart markers (Option A).
//

#include <stdint.h>
#include <string.h>

#include <algorithm>
#include <string>
#include <thread>
#include <vector>

#include "bit_writer.h"
#include "sjpegi.h"

namespace sjpeg {

void Encoder::SinglePassScanMultiThreaded(int num_threads,
                                          int rows_per_interval) {
  if (num_threads > mb_h_) num_threads = mb_h_;
  if (rows_per_interval <= 0) rows_per_interval = 1;

  const int total_intervals =
      (mb_h_ + rows_per_interval - 1) / rows_per_interval;
  const int intervals_per_thread =
      std::max(1, (total_intervals + num_threads - 1) / num_threads);

  struct ThreadChunk {
    std::string data;
    bool ok = true;
  };
  std::vector<ThreadChunk> chunks(num_threads);
  std::vector<std::thread> workers;
  workers.reserve(num_threads);

  const QuantizeBlockFunc quantize_block =
      use_trellis_ ? TrellisQuantizeBlock : quantize_block_;

  for (int t = 0; t < num_threads; ++t) {
    const int start_interval = t * intervals_per_thread;
    const int end_interval =
        std::min(total_intervals, (t + 1) * intervals_per_thread);
    if (start_interval >= total_intervals) {
      chunks[t].ok = true;
      continue;
    }

    workers.emplace_back([this, t, start_interval, end_interval,
                          rows_per_interval, total_intervals, quantize_block,
                          &chunks]() {
      ThreadChunk& chunk = chunks[t];
      StringSink sink(&chunk.data);
      sjpeg::BitWriter thread_bw(&sink);

      const size_t est_size = static_cast<size_t>(mb_w_) *
                              (end_interval - start_interval) *
                              rows_per_interval * 1024;
      const size_t chunk_size =
          std::min<size_t>(256 << 10, std::max<size_t>(4096, est_size));
      chunk.data.reserve(chunk_size);

      RunLevel local_run_levels[64];
      int16_t local_in[64 * 6];
      uint8_t local_rep_buf[4 * 16 * 16];

      for (int interval = start_interval; interval < end_interval; ++interval) {
        int local_DCs[3] = {0, 0, 0};
        const int y_start = interval * rows_per_interval;
        const int y_end = std::min(mb_h_, (interval + 1) * rows_per_interval);

        for (int mb_y = y_start; mb_y < y_end; ++mb_y) {
          for (int mb_x = 0; mb_x < mb_w_; ++mb_x) {
            if (!thread_bw.ReserveMore(2560, chunk_size)) {
              chunk.ok = false;
              return;
            }
            const bool yclip = (mb_y == mb_y_max_);
            GetSamples(mb_x, mb_y, yclip | (mb_x == mb_x_max_), local_in,
                       local_rep_buf);
            fDCT_(local_in, mcu_blocks_);

            int16_t* in = local_in;
            for (int c = 0; c < nb_comps_; ++c) {
              DCTCoeffs base_coeffs;
              for (int i = 0; i < nb_blocks_[c]; ++i) {
                const int dc = quantize_block(in, c, &quants_[quant_idx_[c]],
                                              &base_coeffs, local_run_levels);
                base_coeffs.dc_code_ = GenerateDCDiffCode(dc, &local_DCs[c]);
                CodeBlock(&base_coeffs, local_run_levels, &thread_bw);
                in += 64;
              }
            }
          }
        }

        if (interval < total_intervals - 1) {
          thread_bw.Flush();
          if (!thread_bw.Reserve(2)) {
            chunk.ok = false;
            return;
          }
          const uint8_t rst_marker[2] = {
              0xff, static_cast<uint8_t>(0xd0 + (interval % 8))};
          thread_bw.PutBytes(rst_marker, 2);
        }
      }

      thread_bw.Flush();
      chunk.ok = thread_bw.Finalize();
    });
  }

  for (auto& w : workers) {
    if (w.joinable()) w.join();
  }

  for (int t = 0; t < num_threads; ++t) {
    if (!chunks[t].ok) {
      SetError();
      return;
    }
    if (!chunks[t].data.empty()) {
      if (!bw_.Reserve(chunks[t].data.size())) {
        SetError();
        return;
      }
      bw_.PutBytes(reinterpret_cast<const uint8_t*>(chunks[t].data.data()),
                   chunks[t].data.size());
      std::string().swap(chunks[t].data);  // Free chunk memory immediately
    }
  }
}

void Encoder::SinglePassScanOptimizedMultiThreaded(int num_threads,
                                                   int rows_per_interval) {
  if (num_threads > mb_h_) num_threads = mb_h_;
  if (rows_per_interval <= 0) rows_per_interval = 1;

  const int total_intervals =
      (mb_h_ + rows_per_interval - 1) / rows_per_interval;
  const int intervals_per_thread =
      std::max(1, (total_intervals + num_threads - 1) / num_threads);

  struct ThreadStats {
    uint32_t freq_ac[2][256];
    uint32_t freq_dc[2][12];
  };
  std::vector<ThreadStats> stats(num_threads);
  for (int t = 0; t < num_threads; ++t) {
    memset(stats[t].freq_ac, 0, sizeof(stats[t].freq_ac));
    memset(stats[t].freq_dc, 0, sizeof(stats[t].freq_dc));
  }

  const QuantizeBlockFunc quantize_block =
      use_trellis_ ? TrellisQuantizeBlock : quantize_block_;
  if (use_trellis_) InitCodes(true);

  std::vector<std::thread> histo_workers;
  histo_workers.reserve(num_threads);

  for (int t = 0; t < num_threads; ++t) {
    const int start_interval = t * intervals_per_thread;
    const int end_interval =
        std::min(total_intervals, (t + 1) * intervals_per_thread);
    if (start_interval >= total_intervals) continue;

    histo_workers.emplace_back([this, t, start_interval, end_interval,
                                rows_per_interval, quantize_block, &stats]() {
      RunLevel local_run_levels[64];
      int16_t local_in[64 * 6];
      uint8_t local_rep_buf[4 * 16 * 16];

      for (int interval = start_interval; interval < end_interval; ++interval) {
        int local_DCs[3] = {0, 0, 0};
        const int y_start = interval * rows_per_interval;
        const int y_end = std::min(mb_h_, (interval + 1) * rows_per_interval);

        for (int mb_y = y_start; mb_y < y_end; ++mb_y) {
          for (int mb_x = 0; mb_x < mb_w_; ++mb_x) {
            const bool yclip = (mb_y == mb_y_max_);
            GetSamples(mb_x, mb_y, yclip | (mb_x == mb_x_max_), local_in,
                       local_rep_buf);
            fDCT_(local_in, mcu_blocks_);

            int16_t* in = local_in;
            for (int c = 0; c < nb_comps_; ++c) {
              const int q_idx = quant_idx_[c];
              DCTCoeffs base_coeffs;
              for (int i = 0; i < nb_blocks_[c]; ++i) {
                const int dc = quantize_block(in, c, &quants_[q_idx],
                                              &base_coeffs, local_run_levels);
                base_coeffs.dc_code_ = GenerateDCDiffCode(dc, &local_DCs[c]);
                for (int k = 0; k < base_coeffs.nb_coeffs_; ++k) {
                  const int run = local_run_levels[k].run_;
                  const int tmp = (run >> 4);
                  if (tmp) stats[t].freq_ac[q_idx][0xf0] += tmp;
                  const int suffix = local_run_levels[k].level_;
                  const int sym = ((run & 0x0f) << 4) | (suffix & 0x0f);
                  ++stats[t].freq_ac[q_idx][sym];
                }
                if (base_coeffs.last_ < 63) {
                  ++stats[t].freq_ac[q_idx][0x00];
                }
                const int dc_len = base_coeffs.dc_code_ & 0x0f;
                if (dc_len < 12) {
                  ++stats[t].freq_dc[q_idx][dc_len];
                }
                in += 64;
              }
            }
          }
        }
      }
    });
  }

  for (auto& w : histo_workers) {
    if (w.joinable()) w.join();
  }

  ResetEntropyStats();
  const int nb_tables = (nb_comps_ == 1) ? 1 : 2;
  for (int q = 0; q < nb_tables; ++q) {
    for (int t = 0; t < num_threads; ++t) {
      for (int i = 0; i < 256; ++i) {
        freq_ac_[q][i] += stats[t].freq_ac[q][i];
      }
      for (int i = 0; i < 12; ++i) {
        freq_dc_[q][i] += stats[t].freq_dc[q][i];
      }
    }
  }

  CompileEntropyStats();
  WriteDHT();
  WriteSOS();

  SinglePassScanMultiThreaded(num_threads, rows_per_interval);
}

}  // namespace sjpeg
