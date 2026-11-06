#!/bin/bash
# Writes jobs.txt with one "SAMPLE INPUT_FILE FILE_IDX" line per input file.
cd "$(dirname "${BASH_SOURCE[0]}")"
declare -A SAMPLES=(
    [ttbar]=/ceph/cms/store/user/mmasciov/forClaude/inputs5k/tt
    [qcd]=/ceph/cms/store/user/mmasciov/forClaude/inputs5k/qcd
    [zpmm]=/ceph/cms/store/user/mmasciov/nfs-7/RelValZpToMM_m6000_PU_150X_mcRun4_realistic_v1_STD_D121_RegeneratedGS_PU
)
: > jobs.txt
for s in ttbar qcd zpmm; do
    i=0
    for f in $(ls ${SAMPLES[$s]}/*.root | sort); do
        echo "$s $f $i" >> jobs.txt
        i=$((i+1))
    done
done
wc -l < jobs.txt | xargs echo "jobs:"
