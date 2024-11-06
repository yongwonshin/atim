make clean
clear

NR_DPUS=1 make >/dev/null 2>/dev/null; ./bin/host_code 2>/dev/null; echo ""
NR_DPUS=2 make >/dev/null 2>/dev/null; ./bin/host_code 2>/dev/null; echo ""
NR_DPUS=4 make >/dev/null 2>/dev/null; ./bin/host_code 2>/dev/null; echo ""
NR_DPUS=8 make >/dev/null 2>/dev/null; ./bin/host_code 2>/dev/null; echo ""
NR_DPUS=16 make >/dev/null 2>/dev/null; ./bin/host_code 2>/dev/null; echo ""
NR_DPUS=32 make >/dev/null 2>/dev/null; ./bin/host_code 2>/dev/null; echo ""
NR_DPUS=64 make >/dev/null 2>/dev/null; ./bin/host_code 2>/dev/null; echo ""
NR_DPUS=128 make >/dev/null 2>/dev/null; ./bin/host_code 2>/dev/null; echo ""
NR_DPUS=256 make >/dev/null 2>/dev/null; ./bin/host_code 2>/dev/null; echo ""
NR_DPUS=512 make >/dev/null 2>/dev/null; ./bin/host_code 2>/dev/null; echo ""
NR_DPUS=1024 make >/dev/null 2>/dev/null; ./bin/host_code 2>/dev/null; echo ""
NR_DPUS=2048 make >/dev/null 2>/dev/null; ./bin/host_code 2>/dev/null; echo ""

echo ""
echo "64 / 16"
NR_DPUS=64 NR_TASKLETS=16 BL=8 make >/dev/null 2>/dev/null; ./bin/host_code 2>/dev/null; echo "" #64
NR_DPUS=64 NR_TASKLETS=16 BL=9 make >/dev/null 2>/dev/null; ./bin/host_code 2>/dev/null; echo "" #128
NR_DPUS=64 NR_TASKLETS=16 BL=10 make >/dev/null 2>/dev/null; ./bin/host_code 2>/dev/null; echo "" #256
NR_DPUS=64 NR_TASKLETS=16 BL=11 make >/dev/null 2>/dev/null; ./bin/host_code 2>/dev/null; echo "" #512

echo ""
echo "64 / 8"
NR_DPUS=64 NR_TASKLETS=8 BL=8 make >/dev/null 2>/dev/null; ./bin/host_code 2>/dev/null; echo "" #64
NR_DPUS=64 NR_TASKLETS=8 BL=9 make >/dev/null 2>/dev/null; ./bin/host_code 2>/dev/null; echo "" #128
NR_DPUS=64 NR_TASKLETS=8 BL=10 make >/dev/null 2>/dev/null; ./bin/host_code 2>/dev/null; echo "" #256
NR_DPUS=64 NR_TASKLETS=8 BL=11 make >/dev/null 2>/dev/null; ./bin/host_code 2>/dev/null; echo "" #512

echo ""
echo "64 / 1"
NR_DPUS=64 NR_TASKLETS=1 BL=8 make >/dev/null 2>/dev/null; ./bin/host_code 2>/dev/null; echo ""
NR_DPUS=64 NR_TASKLETS=1 BL=9 make >/dev/null 2>/dev/null; ./bin/host_code 2>/dev/null; echo ""
NR_DPUS=64 NR_TASKLETS=1 BL=10 make >/dev/null 2>/dev/null; ./bin/host_code 2>/dev/null; echo ""
NR_DPUS=64 NR_TASKLETS=1 BL=11 make >/dev/null 2>/dev/null; ./bin/host_code 2>/dev/null; echo ""

echo ""
echo "128 / 16"
NR_DPUS=128 NR_TASKLETS=16 BL=8 make >/dev/null 2>/dev/null; ./bin/host_code 2>/dev/null; echo ""
NR_DPUS=128 NR_TASKLETS=16 BL=9 make >/dev/null 2>/dev/null; ./bin/host_code 2>/dev/null; echo ""
NR_DPUS=128 NR_TASKLETS=16 BL=10 make >/dev/null 2>/dev/null; ./bin/host_code 2>/dev/null; echo ""
NR_DPUS=128 NR_TASKLETS=16 BL=11 make >/dev/null 2>/dev/null; ./bin/host_code 2>/dev/null; echo ""

echo ""
echo "128 / 8"
NR_DPUS=128 NR_TASKLETS=8 BL=8 make >/dev/null 2>/dev/null; ./bin/host_code 2>/dev/null; echo ""
NR_DPUS=128 NR_TASKLETS=8 BL=9 make >/dev/null 2>/dev/null; ./bin/host_code 2>/dev/null; echo ""
NR_DPUS=128 NR_TASKLETS=8 BL=10 make >/dev/null 2>/dev/null; ./bin/host_code 2>/dev/null; echo ""
NR_DPUS=128 NR_TASKLETS=8 BL=11 make >/dev/null 2>/dev/null; ./bin/host_code 2>/dev/null; echo ""

echo ""
echo "128 / 1"
NR_DPUS=128 NR_TASKLETS=1 BL=8 make >/dev/null 2>/dev/null; ./bin/host_code 2>/dev/null; echo ""
NR_DPUS=128 NR_TASKLETS=1 BL=9 make >/dev/null 2>/dev/null; ./bin/host_code 2>/dev/null; echo ""
NR_DPUS=128 NR_TASKLETS=1 BL=10 make >/dev/null 2>/dev/null; ./bin/host_code 2>/dev/null; echo ""
NR_DPUS=128 NR_TASKLETS=1 BL=11 make >/dev/null 2>/dev/null; ./bin/host_code 2>/dev/null; echo ""