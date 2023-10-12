cd ../build
cmake .. -G Ninja
ninja
cd ../custom_gemv
TVM_LOG_DEBUG=DEFAULT=2,src/target/llvm/codegen_llvm.cc=-1,src/arith/int_set.cc=-1 python3 gemv.py