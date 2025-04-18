/***************************************************************************************************
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed,
 * transmitted, transcribed, stored in a retrieval system, or translated into any human
 * or computer language in any form by any means,electronic, mechanical, manual or otherwise,
 * or disclosed to third parties without the express written permission of Samsung Electronics.
 * (Use of the Software is restricted to non-commercial, personal or academic, research purpose
 * only)
 **************************************************************************************************/

#include <string>
#include <vector>

#include "AddressMapping.h"
#include "tools/emulator_api/PimSimulator2.h"

PimSimulator2::PimSimulator2() : bst_size_(16), cycle_(0) {}

PimSimulator2::~PimSimulator2() {}

void PimSimulator2::initialize(const string& device_ini_file_name,
                               const string& system_ini_file_name, size_t megs_of_memory,
                               size_t num_pim_chan, size_t num_pim_rank) {
  mem_ = make_shared<MultiChannelMemorySystem>(device_ini_file_name, system_ini_file_name, ".",
                                               "example_app", megs_of_memory);
  addr_mapping_ = mem_->addrMapping;
  pim_kernel_ = make_shared<PIMKernel>(mem_, num_pim_chan, num_pim_rank);
}

void PimSimulator2::deinitialize() {}

void PimSimulator2::execute_kernel(void* trace_data, size_t num_trace) {
  vector<TraceDataBst> trace_bst;
  convert_to_burst_trace(trace_data, &trace_bst, num_trace);
  push_trace(&trace_bst);
  run();
  trace_bst.clear();
}

void PimSimulator2::convert_to_burst_trace(void* trace_data, vector<TraceDataBst>* trace_bst,
                                           size_t num_trace) {
  TraceDataBst tmp;
  MemTraceData* vec_trace_data = static_cast<MemTraceData*>(trace_data);
  unsigned channelNumber, rank, bank, row, col;

  for (int i = 0; i < num_trace; i++) {
    tmp.cmd = vec_trace_data[i].cmd;
    tmp.addr = vec_trace_data[i].addr;
    if (vec_trace_data[i].cmd == 'B') {
      tmp.ch = vec_trace_data[i].block_id;
    } else {
      addr_mapping_->addressMapping(vec_trace_data[i].addr, channelNumber, rank, bank, row, col);
      // cout << "ch : " << channelNumber << " ra : " << rank << " ba : " << bank
      //      << " row : " << row << " col : " <<col <<endl;
      if (row >= 16384) {
        cout << "overflow" << endl;
      }
      if (channelNumber >= 64) {
        cout << "overflow" << endl;
      }
      if (bank >= 16) {
        cout << "overflow" << endl;
      }
      if (col >= 32) {
        cout << "overflow" << endl;
      }
      tmp.ch = channelNumber;
    }
    memcpy(tmp.data.u16Data_, vec_trace_data[i].data, sizeof(uint16_t) * bst_size_);

    trace_bst->push_back(tmp);
  }
}

void PimSimulator2::convert_arr_to_burst(void* data, size_t data_size, BurstType* bst) {
  uint16_t* fp16_data = static_cast<uint16_t*>(data);
  for (int i = 0; i < (data_size / sizeof(uint16_t)); i += bst_size_) {
    bst[i / bst_size_].set(fp16_data[i], fp16_data[i + 1], fp16_data[i + 2], fp16_data[i + 3],
                           fp16_data[i + 4], fp16_data[i + 5], fp16_data[i + 6], fp16_data[i + 7],
                           fp16_data[i + 8], fp16_data[i + 9], fp16_data[i + 10], fp16_data[i + 11],
                           fp16_data[i + 12], fp16_data[i + 13], fp16_data[i + 14],
                           fp16_data[i + 15]);
  }
}

void PimSimulator2::push_trace(vector<TraceDataBst>* trace_bst) {
  if (trace_bst->size() < 1) {
    cout << "there is no trace file" << endl;
  }

  bool is_write = false;
  for (int i = 0; i < trace_bst->size(); i++) {
    if ((*trace_bst)[i].cmd == 'B') {
      mem_->addBarrier((*trace_bst)[i].ch);
      continue;
    }

    if ((*trace_bst)[i].cmd == 'R') {
      is_write = false;
    } else {
      is_write = true;
    }

    mem_->addTransaction(is_write, (*trace_bst)[i].addr, "tag", &(*trace_bst)[i].data);
  }
}
/*
void PimSimulator2::enqueue_trace(void* trace_data, size_t num_trace, burst_data* burst)
{
    MemTraceData* vec_trace_data = static_cast<MemTraceData*>(trace_data);
    bool is_write = false;
    for (int i=0; i<num_trace; i++)
    {
        if ( vec_trace_data[i].cmd == 'B' )
        {
            mem_->addBarrier(vec_trace_data[i].block_id);
            continue;
        }
        if (vec_trace_data[i].cmd  == 'R')
        {
            is_write = false;
        }
        else
        {
            is_write = true;
        }
        memcpy(burst[i].u16_data_, vec_trace_data[i].data, sizeof(uint16_t)*bst_size_);
        mem_->addTransaction(is_write, vec_trace_data[i].addr, "tag", &burst[i]);
    }
}
*/
void PimSimulator2::preload_data_with_addr(uint64_t addr, void* data, size_t data_size) {
  int num_burst = (data_size / sizeof(uint16_t)) / bst_size_;
  BurstType* buffer_burst = new BurstType[num_burst];
  convert_arr_to_burst(data, data_size, buffer_burst);

  for (int i = 0; i < (data_size / sizeof(uint16_t)) / bst_size_; i++) {
    mem_->addTransaction(true, addr + i * 32, &buffer_burst[i]);
  }
  run();

  delete[] buffer_burst;
}

void PimSimulator2::run() {
  while (mem_->hasPendingTransactions()) {
    cycle_++;
    mem_->update();
  }
}

void PimSimulator2::read_result(uint16_t* output_data, uint64_t addr, size_t data_size) {
  int num_burst = (data_size / sizeof(uint16_t)) / bst_size_;
  BurstType* output_burst = new BurstType[num_burst];

  for (int i = 0; i < num_burst; i++) {
    mem_->addTransaction(false, addr + i * 32, "output", &output_burst[i]);
  }

  run();

  for (int i = 0; i < num_burst; i++) {
    for (int j = 0; j < bst_size_; j++) {
      output_data[i * bst_size_ + j] = output_burst[i].u16Data_[j];
    }
  }

  delete[] output_burst;
}

void PimSimulator2::read_result_gemv(uint16_t* output_data, uint64_t addr, size_t data_dim) {
  BurstType* buffer_burst = new BurstType[data_dim];

  pim_kernel_->readResult(buffer_burst, pimBankType::ODD_BANK, data_dim, addr);
  run();

  fp16* output_fp16 = (fp16*)output_data;
  for (int i = 0; i < data_dim; i++) {
    output_fp16[i] = buffer_burst[i].fp16ReduceSum();
  }
  delete[] buffer_burst;
}

void PimSimulator2::read_result_gemv_tree(uint16_t* output_data, uint64_t addr, size_t output_dim,
                                          size_t batch_dim, int num_input_tile) {
  BurstType* buffer_burst = new BurstType[output_dim * batch_dim * num_input_tile];
  pim_kernel_->readResult(buffer_burst, pimBankType::ODD_BANK,
                          output_dim * batch_dim * num_input_tile, addr);
  run();

  fp16* output_fp16 = (fp16*)output_data;
  fp16* temp_fp16 = new fp16[num_input_tile];

  for (int b = 0; b < batch_dim; b++) {
    for (int i = 0; i < output_dim; i++) {
      pim_kernel_->adderTree(&buffer_burst[b * output_dim * num_input_tile + i], output_dim,
                             num_input_tile, 0, temp_fp16);
      output_fp16[b * output_dim + i] = temp_fp16[0];
    }
  }
  delete[] temp_fp16;
  delete[] buffer_burst;
}

size_t PimSimulator2::get_cycle() { return cycle_; }

void PimSimulator2::reset_cycle() { cycle_ = 0; }
