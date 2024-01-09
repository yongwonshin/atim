////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2015, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

// A simple best fit memory allocator with eager compaction.  Manages block
// sub-allocation.
// For use when memory efficiency is more important than allocation speed.
// O(log n) time.

#ifndef TVM_RUNTIME_OPENCL_HBMPIM_PIM_LIBRARY_SIMPLE_HEAP_H_
#define TVM_RUNTIME_OPENCL_HBMPIM_PIM_LIBRARY_SIMPLE_HEAP_H_

#include <tvm/runtime/logging.h>

#include <cmath>
#include <deque>
#include <map>
#include <utility>

#include "utils.h"

namespace tvm {
namespace runtime {
namespace pim_library {

template <typename Allocator>
class SimpleHeap {
 private:
  struct Fragment_T {
    typedef std::multimap<size_t, uintptr_t>::iterator ptr_t;
    ptr_t free_list_entry_;
    size_t size;

    Fragment_T(ptr_t Iterator, size_t Len) : free_list_entry_(Iterator), size(Len) {}
    Fragment_T() = default;
  };

  struct Block {
    uintptr_t base_ptr_;
    size_t length_;

    Block(uintptr_t base, size_t length) : base_ptr_(base), length_(length) {}
    Block() = default;
  };

  Allocator block_allocator_;
  void* pim_base_ = nullptr;

  std::multimap<size_t, uintptr_t> free_list_;
  std::map<uintptr_t, std::map<uintptr_t, Fragment_T>> block_list_;
  std::deque<Block> block_cache_;

  size_t in_use_size_;
  size_t cache_size_;
  size_t align_size_;

  __forceinline size_t get_aligned_bytes(size_t bytes) {
    return (align_size_ * ceil((float)bytes / align_size_));
  }
  __forceinline bool isFree(const Fragment_T& node) {
    return node.free_list_entry_ != free_list_.end();
  }
  __forceinline void setUsed(Fragment_T& node) { node.free_list_entry_ = free_list_.end(); }
  __forceinline void setFree(Fragment_T& node, typename Fragment_T::ptr_t Iterator) {
    node.free_list_entry_ = Iterator;
  }
  __forceinline Fragment_T makeFragment(size_t Len) { return Fragment_T(free_list_.end(), Len); }
  __forceinline Fragment_T makeFragment(typename Fragment_T::ptr_t Iterator, size_t Len) {
    return Fragment_T(Iterator, Len);
  }

  cl_context context_;

 public:
  explicit SimpleHeap(const Allocator& BlockAllocator = Allocator())
      : block_allocator_(BlockAllocator),
        in_use_size_(0),
        cache_size_(0),
        align_size_(2 * 1024 * 1024) {}
  ~SimpleHeap() {
    trim();
    // Leak here may be due to the user.  Check is for debugging only.
    // ICHECK(in_use_size_ == 0) << "Leak in SimpleHeap.";
  }

  SimpleHeap(const SimpleHeap& rhs) = delete;
  SimpleHeap(SimpleHeap&& rhs) = delete;
  SimpleHeap& operator=(const SimpleHeap& rhs) = delete;
  SimpleHeap& operator=(SimpleHeap&& rhs) = delete;

  void set_context(cl_context context) { context_ = context; }

  void set_pim_base(void* pim_base) { pim_base_ = pim_base; }
  void* get_pim_base() { return pim_base_; }
  void* alloc(size_t bytes, Device dev) {
    size_t aligned_bytes = get_aligned_bytes(bytes);

    if (aligned_bytes > max_alloc()) {
      LOG(FATAL) << "Requested allocation is larger than block size.";
      return nullptr;
    }

    // Find best fit.
    auto free_fragment = free_list_.lower_bound(aligned_bytes);
    uintptr_t base;
    size_t size;

    if (free_fragment != free_list_.end()) {
      base = free_fragment->second;
      size = free_fragment->first;
      free_list_.erase(free_fragment);

      ICHECK(size >= aligned_bytes) << "SimpleHeap: map lower_bound failure.";

      // Find the containing block and fragment
      auto it = block_list_.upper_bound(base);
      it--;
      auto& frag_map = it->second;
      const auto& fragment = frag_map.find(base);

      ICHECK(fragment != frag_map.end()) << "Inconsistency in SimpleHeap.";
      ICHECK(size == fragment->second.size) << "Inconsistency in SimpleHeap.";

      // Sub-allocate from fragment.
      fragment->second.size = aligned_bytes;
      setUsed(fragment->second);
      // Record remaining free space.
      if (size > aligned_bytes) {
        free_fragment =
            free_list_.insert(std::make_pair(size - aligned_bytes, base + aligned_bytes));
        frag_map[base + aligned_bytes] = makeFragment(free_fragment, size - aligned_bytes);
      }
      return reinterpret_cast<void*>(base);
    }

    // No usable fragment, check block cache
    if (!block_cache_.empty()) {
      const auto& block = block_cache_.back();
      base = block.base_ptr_;
      size = block.length_;
      block_cache_.pop_back();
      cache_size_ -= size;
    } else {  // Alloc new block
      if (dev.device_id < 0) {
        LOG(FATAL) << "device id for allocating PIM memory is invalid ";
      }
      void* ptr = block_allocator_.alloc(context_, aligned_bytes, size, dev);
      set_pim_base(block_allocator_.get_pim_base());

      base = reinterpret_cast<uintptr_t>(ptr);
      if (ptr == nullptr) return ptr;
    }

    in_use_size_ += size;
    ICHECK(size >= aligned_bytes) << "Alloc exceeds block size.";
    // Sub alloc and insert free region.
    if (size > aligned_bytes) {
      free_fragment = free_list_.insert(std::make_pair(size - aligned_bytes, base + aligned_bytes));
      block_list_[base][base + aligned_bytes] = makeFragment(free_fragment, size - aligned_bytes);
    }
    // Track used region
    block_list_[base][base] = makeFragment(aligned_bytes);

    return reinterpret_cast<void*>(base);
  }

  bool free(void* ptr) {
    if (ptr == nullptr) return true;

    uintptr_t base = reinterpret_cast<uintptr_t>(ptr);

    // Find fragment and validate.
    auto frag_map_it = block_list_.upper_bound(base);
    if (frag_map_it == block_list_.begin()) return false;
    frag_map_it--;
    auto& frag_map = frag_map_it->second;
    auto fragment = frag_map.find(base);
    if (fragment == frag_map.end() || isFree(fragment->second)) return false;

    // Merge lower
    if (fragment != frag_map.begin()) {
      auto lower = fragment;
      lower--;
      if (isFree(lower->second)) {
        free_list_.erase(lower->second.free_list_entry_);
        lower->second.size += fragment->second.size;
        frag_map.erase(fragment);
        fragment = lower;
      }
    }

    // Merge upper
    {
      auto upper = fragment;
      upper++;
      if ((upper != frag_map.end()) && isFree(upper->second)) {
        free_list_.erase(upper->second.free_list_entry_);
        fragment->second.size += upper->second.size;
        frag_map.erase(upper);
      }
    }

    // Move whole free blocks to block cache
    if (frag_map.size() == 1) {
      in_use_size_ -= fragment->second.size;
      cache_size_ += fragment->second.size;
      block_cache_.push_back(Block(fragment->first, fragment->second.size));
      block_list_.erase(frag_map_it);

      // Release old blocks when over cache limit.
      while ((block_cache_.size() > 1) && (cache_size_ > in_use_size_ * 2)) {
        const auto& block = block_cache_.front();
        block_allocator_.free(reinterpret_cast<void*>(block.base_ptr_), block.length_);
        cache_size_ -= block.length_;
        block_cache_.pop_front();
      }

      // Don't publish free space since block was moved to the cache.
      return true;
    }

    // Report free fragment
    const auto& freeEntry =
        free_list_.insert(std::make_pair(fragment->second.size, fragment->first));
    setFree(fragment->second, freeEntry);

    return true;
  }

  void trim() {
    for (const auto& block : block_cache_)
      block_allocator_.free(reinterpret_cast<void*>(block.base_ptr_), block.length_);
    block_cache_.clear();
    cache_size_ = 0;
  }

  size_t max_alloc() const { return block_allocator_.block_size(); }

  bool get_pim_alloc_done(int device_id) { return block_allocator_.pim_alloc_done[device_id]; }
  uint64_t get_g_pim_base_addr(int device_id) {
    return block_allocator_.g_pim_base_addr[device_id];
  }
};

}  // namespace pim_library
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_OPENCL_HBMPIM_PIM_LIBRARY_SIMPLE_HEAP_H_
