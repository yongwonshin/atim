/*
 * Copyright (C) 2022 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed,
 * transmitted, transcribed, stored in a retrieval system or translated into any human or computer
 * language in any form by any means, electronic, mechanical, manual or otherwise, or disclosed to
 * third parties without the express written permission of Samsung Electronics. (Use of the Software
 * is restricted to non-commercial, personal or academic, research purpose only)
 */

#include <cmath>

#include "../hbmpim_common.h"

namespace tvm {
namespace runtime {
namespace pim_library {

PimCrfBinGen::PimCrfBinGen(cl::HBMPIMWorkspace* w)
    : w_(w), is_gemv_tile_tree_(true), max_crf_size_(128), pbi_(&vega20_pbi) {}

PimCrfBinGen::~PimCrfBinGen() {
  for (auto it = crf_lut_.begin(); it != crf_lut_.end(); it++) {
    // pim_manager_->free_memory((void*)it->second, MEM_TYPE_DEVICE);
  }
  crf_lut_.clear();
}

int PimCrfBinGen::GetMaxCrfSize() { return max_crf_size_; }

void PimCrfBinGen::GenBinaryWithLoop(PimOpType op_type, int lc, uint8_t* bin_buf, int* crf_sz) {
  VLOG(2) << "[START] " << __FUNCTION__ << " called";
  createPimCmd(op_type, lc);
  changeToBinary(bin_buf, crf_sz);
  VLOG(2) << "[END] " << __FUNCTION__ << " called";
}

void PimCrfBinGen::createPimCmd(PimOpType op_type, int lc) {
  VLOG(2) << "[START] " << __FUNCTION__ << " called";

  if (op_type == OP_ELT_ADD) {
    std::vector<PimCommand> tmp_cmds{
            PimCommand(PimCmdType::FILL, PimOpdType::GRF_A, PimOpdType::EVEN_BANK),
            PimCommand(PimCmdType::ADD, PimOpdType::GRF_A, PimOpdType::GRF_A, PimOpdType::EVEN_BANK, 1),
            PimCommand(PimCmdType::NOP, 23), PimCommand(PimCmdType::FILL, PimOpdType::GRF_B, PimOpdType::ODD_BANK),
            PimCommand(PimCmdType::ADD, PimOpdType::GRF_B, PimOpdType::GRF_B, PimOpdType::ODD_BANK, 1),
            PimCommand(PimCmdType::NOP, 23) /*,
            PimCommand(PimCmdType::NOP, 0)*/};
    cmds_.assign(tmp_cmds.begin(), tmp_cmds.end());
  } else if (op_type == OP_ELT_MUL) {
    std::vector<PimCommand> tmp_cmds{
            PimCommand(PimCmdType::FILL, PimOpdType::GRF_A, PimOpdType::EVEN_BANK),
            PimCommand(PimCmdType::MUL, PimOpdType::GRF_A, PimOpdType::GRF_A, PimOpdType::EVEN_BANK, 1),
            PimCommand(PimCmdType::NOP, 23), PimCommand(PimCmdType::FILL, PimOpdType::GRF_B, PimOpdType::ODD_BANK),
            PimCommand(PimCmdType::MUL, PimOpdType::GRF_B, PimOpdType::GRF_B, PimOpdType::ODD_BANK, 1),
            PimCommand(PimCmdType::NOP, 23) /*,
            PimCommand(PimCmdType::NOP, 0)*/};
    cmds_.assign(tmp_cmds.begin(), tmp_cmds.end());
  } else if (op_type == OP_RELU) {
    std::vector<PimCommand> tmp_cmds{
        PimCommand(PimCmdType::FILL, PimOpdType::GRF_A, PimOpdType::EVEN_BANK, 1, 0, 0, 0, 1),
        PimCommand(PimCmdType::NOP, 15),
        PimCommand(PimCmdType::FILL, PimOpdType::GRF_B, PimOpdType::ODD_BANK, 1, 0, 0, 0, 1),
        PimCommand(PimCmdType::NOP, 15) /*, PimCommand(PimCmdType::NOP, 0)*/};
    cmds_.assign(tmp_cmds.begin(), tmp_cmds.end());
  } else if (op_type == OP_COPY) {
    std::vector<PimCommand> tmp_cmds{
        PimCommand(PimCmdType::FILL, PimOpdType::GRF_A, PimOpdType::EVEN_BANK, 1, 0, 0, 0, 0),
        PimCommand(PimCmdType::NOP, 15),
        PimCommand(PimCmdType::FILL, PimOpdType::GRF_B, PimOpdType::ODD_BANK, 1, 0, 0, 0, 0),
        PimCommand(PimCmdType::NOP, 15) /*, PimCommand(PimCmdType::NOP, 0)*/};
    cmds_.assign(tmp_cmds.begin(), tmp_cmds.end());
  } else if (op_type == OP_GEMV) {
    if (is_gemv_tile_tree_) {
      std::vector<PimCommand> tmp_cmds{
          PimCommand(PimCmdType::MAC, PimOpdType::GRF_B, PimOpdType::GRF_A, PimOpdType::EVEN_BANK,
                     1, 0, 0, 0),
          PimCommand(PimCmdType::JUMP, 7, 2),
          PimCommand(PimCmdType::NOP, 23),
          PimCommand(PimCmdType::MAC, PimOpdType::GRF_B, PimOpdType::GRF_A, PimOpdType::ODD_BANK, 1,
                     0, 0, 0),
          PimCommand(PimCmdType::JUMP, 7, 2),
          PimCommand(PimCmdType::NOP, 23),
      };
      cmds_.assign(tmp_cmds.begin(), tmp_cmds.end());
    } else {
      int even_lc = 8 * ceil((float)lc / 2) - 1;
      int odd_lc = 8 * (lc / 2) - 1;
      std::vector<PimCommand> tmp_cmds{
          PimCommand(PimCmdType::MAC, PimOpdType::GRF_B, PimOpdType::GRF_A, PimOpdType::EVEN_BANK,
                     1, 0, 0, 0),
          PimCommand(PimCmdType::JUMP, even_lc, 2),
          PimCommand(PimCmdType::MAC, PimOpdType::GRF_B, PimOpdType::GRF_A, PimOpdType::ODD_BANK, 1,
                     0, 0, 0),
          PimCommand(PimCmdType::JUMP, odd_lc, 2), PimCommand(PimCmdType::NOP, 23)};
      cmds_.assign(tmp_cmds.begin(), tmp_cmds.end());
    }
  } else if (op_type == OP_BN) {
    std::vector<PimCommand> tmp_cmds{
        PimCommand(PimCmdType::MAD, PimOpdType::GRF_A, PimOpdType::EVEN_BANK, PimOpdType::SRF_M,
                   PimOpdType::SRF_A, 1, 0, 0, 0),
        PimCommand(PimCmdType::NOP, 7),
        PimCommand(PimCmdType::MAD, PimOpdType::GRF_A, PimOpdType::GRF_A, PimOpdType::SRF_M,
                   PimOpdType::SRF_A, 1, 0, 0, 1),
        PimCommand(PimCmdType::NOP, 7),
        PimCommand(PimCmdType::MAD, PimOpdType::GRF_B, PimOpdType::ODD_BANK, PimOpdType::SRF_M,
                   PimOpdType::SRF_A, 1, 0, 0, 0),
        PimCommand(PimCmdType::NOP, 7),
        PimCommand(PimCmdType::MAD, PimOpdType::GRF_B, PimOpdType::GRF_B, PimOpdType::SRF_M,
                   PimOpdType::SRF_A, 1, 0, 0, 1),
        PimCommand(PimCmdType::NOP, 23)
        /*PimCommand(PimCmdType::NOP, 0)*/};
    cmds_.assign(tmp_cmds.begin(), tmp_cmds.end());
  }

  if ((lc != 0 && is_gemv_tile_tree_) == true || op_type != OP_GEMV) {
    cmds_.push_back(PimCommand(PimCmdType::JUMP, lc, cmds_.size() + 1));
  }

  cmds_.push_back(PimCommand(PimCmdType::EXIT, 0));

  int nop_cnt = 8 - cmds_.size() % 8;
  for (int i = 0; i < nop_cnt; i++) cmds_.push_back(PimCommand(PimCmdType::NOP, 0));
  VLOG(2) << "[END] " << __FUNCTION__ << " called";
}

void PimCrfBinGen::changeToBinary(uint8_t* crf_binary, int* crf_size) {
  VLOG(2) << "[START] " << __FUNCTION__ << " called";
  PimCommand nop_cmd(PimCmdType::NOP, 0);
  *crf_size = cmds_.size() * sizeof(uint32_t);

  for (int i = 0; i < cmds_.size(); i++) {
    uint32_t u32_data_ = cmds_[i].to_int();
    memcpy(&crf_binary[i * 4], &u32_data_, sizeof(uint32_t));
  }
  VLOG(2) << "[END] " << __FUNCTION__ << " called";
}

void PimCrfBinGen::SetGemvTileTree(bool is_gemv_tile_tree) {
  is_gemv_tile_tree_ = is_gemv_tile_tree;
}

int PimCrfBinGen::GetLoopCounter(PimOpType op_type, int input_size) {
  int lc = 0;
  int num_transaction = (input_size / 16) / sizeof(uint16_t);
  int num_parallelism =
      pbi_->num_pim_blocks * pbi_->num_pim_chan * pbi_->num_pim_rank * pbi_->num_grf;
  int num_tile = num_transaction / num_parallelism;

  if (op_type == OP_GEMV) {
    if (is_gemv_tile_tree_)
      lc = (input_size / pbi_->trans_size / pbi_->num_grf_A / 2) - 1;
    else
      lc = input_size / pbi_->trans_size / pbi_->num_grf_A;
  } else
    lc = num_tile / 2 - 1;
  return lc;
}

void* PimCrfBinGen::FindCrf(PimOpType op_type, int data_size) {
  VLOG(2) << "[START] " << __FUNCTION__ << " called";
  void* addr = nullptr;

  auto found = crf_lut_.find(std::make_pair(op_type, data_size));
  if (found != crf_lut_.end()) {
    addr = found->second;
  }

  VLOG(2) << "[END] " << __FUNCTION__ << " called";
  return addr;
}

void PimCrfBinGen::InsertToCrfLUT(PimOpType op_type, int data_size, void* data) {
  crf_lut_.insert(std::make_pair(std::make_pair(op_type, data_size), data));
}

}  // namespace pim_library
}  // namespace runtime
}  // namespace tvm
