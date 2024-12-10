sed -i "s/dpu_number = [0-9]\+;/dpu_number = $2;/" Param.h
sed -i "s/nr_elements = [0-9]\+;/nr_elements = $1;/" Param.h
make > /dev/null 2> /dev/null
./bin/host
echo ""

# ./test.sh 6553600	1
# ./test.sh 26214400	4
# ./test.sh 104857600	16
# ./test.sh 419430400	64
# ./test.sh 6553600	1
# ./test.sh 6553600	4
# ./test.sh 6553600	16
# ./test.sh 6553600	64
# ./test.sh 400000000	256
# ./test.sh 400000000	512
# ./test.sh 400000000	1024
# ./test.sh 400000000	2048