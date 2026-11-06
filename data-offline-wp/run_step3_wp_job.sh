#!/bin/bash
#
# Offline step3 + trackingNtuple in CMSSW_20_1_0_pre4 (+ cms-sw#52011, #52015, MkFit data e88cb92,
# offline track-purity DNN) with the DNN cuts switched off, for working-point tuning.
# Arguments: SAMPLE INPUT_FILE FILE_IDX
#
set -e
set -o pipefail

SAMPLE=$1
INPUT_FILE=$2
FILE_IDX=$3
CMSSW_AREA=/ceph/users/jchismar/CMSSW_20_1_0_pre4/src
OUTPUT_DIR=${OUTPUT_DIR:-root://redirector.t2.ucsd.edu:1095//store/user/jachisma/trackingNtuples_offline_20_1_0_pre4_wp/${SAMPLE}}
OUTPUT_FILE="trackingNtuple_${SAMPLE}_${FILE_IDX}.root"
NTHREADS=${NTHREADS:-4}
export OMP_NUM_THREADS=1   # otherwise OpenMP starts one thread per core in every stream

echo "Sample: $SAMPLE  Input: $INPUT_FILE  Index: $FILE_IDX  Host: $(hostname)  Date: $(date)"

source /cvmfs/cms.cern.ch/cmsset_default.sh
CURRENT_DIR=$PWD
cd $CMSSW_AREA && eval $(scram runtime -sh) && cd $CURRENT_DIR
echo "CMSSW_BASE=$CMSSW_BASE"

WORK_DIR=$(mktemp -d)
cd $WORK_DIR
CFG_FILE="step3_wp_${SAMPLE}_${FILE_IDX}_cfg.py"

# Inputs already have PU200 mixed in (GEN-SIM-DIGI-RAW), so no --pileup here.
cmsDriver.py step3 \
    --conditions auto:phase2_realistic_T35 \
    --geometry ExtendedRun4D121 \
    --era Phase2C22I13M9 \
    --datatier DQMIO \
    --eventcontent DQM \
    -n ${NEVENTS:--1} \
    --filein file:${INPUT_FILE} \
    --fileout file:step3.root \
    --python_filename ${CFG_FILE} \
    --nThreads ${NTHREADS} \
    --accelerators cpu \
    -s RAW2DIGI,RECO:reconstruction_trackingOnly,VALIDATION:@trackingOnlyValidation,DQM:@trackingOnlyDQM \
    --procModifiers trackingIters01,trackingMkFitHighPtTripletStep,seedingLST,trackingLST \
    --customise Validation/RecoTrack/customiseTrackingNtuple.customiseTrackingNtuple \
    --customise_command="process.trackingNtuple.clusterMasks = []; process.trackingNtuple.seedAlgoDetect = cms.untracked.bool(False); process.trackingNtuple.seedAlgos = cms.untracked.vuint32(4); process.highPtTripletStepSelector.qualityCutsPrompt = [-1., -1., -1.]; process.highPtTripletStepSelector.qualityCutsDisplaced = [-1., -1., -1.]; process.highPtTripletStepSelector.qualityCutsPromptHighPt = [-1., -1., -1.]; process.highPtTripletStepSelector.qualityCutsDisplacedHighPt = [-1., -1., -1.]" \
    --no_exec

# Fail early if the DNN is not the highPtTriplet selector (e.g. a missing procModifier)
python3 - "$CFG_FILE" <<'PY'
import sys, importlib.util
spec = importlib.util.spec_from_file_location("cfg", sys.argv[1]); cfg = importlib.util.module_from_spec(spec); spec.loader.exec_module(cfg)
sel = cfg.process.highPtTripletStepSelector
assert sel.type_() == "TrackTorchClassifierFromSoA", f"highPtTripletStepSelector is {sel.type_()}, not the DNN"
cuts = ("qualityCutsPrompt", "qualityCutsDisplaced", "qualityCutsPromptHighPt", "qualityCutsDisplacedHighPt")
assert all(list(getattr(sel, c, [-1.0] * 3)) == [-1.0] * 3 for c in cuts), "cuts not switched off"
print("Check OK: highPtTripletStepSelector is the DNN with cuts switched off")
PY

cmsRun ${CFG_FILE} 2>&1 | tee cmsRun.log

if [ -f trackingNtuple.root ]; then
    export GFAL_PYTHONBIN="/usr/bin/python3"
    if [[ $OUTPUT_DIR == root://* ]]; then
        gfal-copy -f trackingNtuple.root ${OUTPUT_DIR}/${OUTPUT_FILE}
    else
        mkdir -p ${OUTPUT_DIR} && cp trackingNtuple.root ${OUTPUT_DIR}/${OUTPUT_FILE}
    fi
    echo "Output copied to ${OUTPUT_DIR}/${OUTPUT_FILE}"
else
    echo "ERROR: trackingNtuple.root not created"
    exit 1
fi

cd /tmp && rm -rf $WORK_DIR
echo "Job finished: $(date)"
