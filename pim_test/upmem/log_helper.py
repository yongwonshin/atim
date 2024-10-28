import re
import difflib

def extract_section(filename):
    """주어진 파일에서 barrier_wait부터 [LLVM Source]까지의 섹션을 추출"""
    with open(filename, 'r') as f:
        lines = f.readlines()

    # barrier_wait가 나오는 줄 찾기
    barrier_start = -1
    llvm_source_start = -1

    for i, line in enumerate(lines):
        if 'barrier_wait' in line:
            barrier_start = i
        elif '[LLVM Source]' in line:
            llvm_source_start = i
            break

    if barrier_start != -1 and llvm_source_start != -1:
        # barrier_wait부터 [LLVM Source] - 4번째 줄까지
        return lines[barrier_start:llvm_source_start - 3]
    else:
        return None

def compare_files_with_diff(file1, file2):
    """두 파일의 추출된 부분을 Ndiff 스타일로 비교"""
    section1 = extract_section(file1)
    section2 = extract_section(file2)

    if section1 is None or section2 is None:
        print("추출할 수 있는 섹션이 없습니다.")
        return

    # Ndiff 방식으로 차이점 출력
    for line1, line2 in zip(section1, section2):
        print(f"{line1.rstrip():<120} | {line2.rstrip():<120}")

# 파일 경로 설정
profile = "opt"
i1 = 1  # i1 값 설정
i2 = 2  # i2 값 설정
j = 5   # j 값 설정

file1 = f"./results/{profile}{i1}_{j:02d}_gemvRCTile.txt.txt"
file2 = f"./results/{profile}{i2}_{j:02d}_gemvRCTile.txt.txt"

# 파일 비교 실행
compare_files_with_diff(file1, file2)