import math
import sys

org_arr=[]
for line in open(sys.argv[1],"r"):
    segs = line.strip().split("\t")
    org_arr.append((int(segs[0]),float(segs[1])))

def calc():
    global org_arr

    idx = 0

    rank = 0
    M = 0
    N = 0

    arr = sorted(org_arr,key=lambda p: p[1])

    count = len(arr)
    
    while idx < count:
        if arr[idx][0] == 0:
            N = N + 1
        elif arr[idx][0] == 1:
            M = M + 1
            rank = rank + idx + 1
        
        idx += 1

    auc = (rank*1.0 - M*(M+1)/2.0) / ( M * N * 1.0 )
 
    print auc

calc()
