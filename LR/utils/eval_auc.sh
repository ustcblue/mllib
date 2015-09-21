sort -n -k $2 $1 | awk -F'\t' '{if($1==0){N=N+1} if($1>0){M=M+1;S=S+NR}}END{print N,M,S,(S-M*(M+1)/2)/(M*N)}'
