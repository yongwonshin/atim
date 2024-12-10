cd va

rm -rf ./bin
NR_DPUS=2048 L=67108864 make > /dev/null 2> /dev/null
./bin/host | grep -E "initial CPU-DPU input transfer|DPU Kernel Time|DPU-CPU Time" | awk \'{printf "%s\t", $NF}\'\
rm -rf ./bin
NR_DPUS=1536 L=67108864 make > /dev/null 2> /dev/null
./bin/host | grep -E "initial CPU-DPU input transfer|DPU Kernel Time|DPU-CPU Time" | awk \'{printf "%s\t", $NF}\'\
rm -rf ./bin
NR_DPUS=1024 L=67108864 make > /dev/null 2> /dev/null
./bin/host | grep -E "initial CPU-DPU input transfer|DPU Kernel Time|DPU-CPU Time" | awk \'{printf "%s\t", $NF}\'\

cd ../red
rm -rf ./bin
NR_DPUS=2048 L=67108864 make > /dev/null 2> /dev/null
./bin/host
rm -rf ./bin
NR_DPUS=1536 L=67108864 make > /dev/null 2> /dev/null
./bin/host
rm -rf ./bin
NR_DPUS=1024 L=67108864 make > /dev/null 2> /dev/null
./bin/host