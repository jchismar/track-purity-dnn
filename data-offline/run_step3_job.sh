#!/bin/bash
#
# Wrapper script to run cmsDriver for a single file with 100 events
# Arguments: DATASET DATASET_NAME INPUT_FILE FILE_IDX
#

set -e  # Exit on error

DATASET=$1
DATASET_NAME=$2
INPUT_FILE=$3
FILE_IDX=$4

echo "=================================================="
echo "Starting CMS job"
echo "Dataset: $DATASET"
echo "Dataset Name: $DATASET_NAME"
echo "Input File: $INPUT_FILE"
echo "File Index: $FILE_IDX"
echo "Working Directory: $(pwd)"
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "=================================================="

CURRENT_DIR=$PWD

source /cvmfs/cms.cern.ch/cmsset_default.sh
CMSSW_VERSION=CMSSW_17_0_0_pre2

cd /ceph/users/jchismar/${CMSSW_VERSION}/src/
cmsenv
echo "CMSSW_BASE=$CMSSW_BASE"
cd $CURRENT_DIR

WORK_DIR=$(mktemp -d)
cd $WORK_DIR
echo "Working in temporary directory: $WORK_DIR"

OUTPUT_FILE="output_step3_${DATASET_NAME}_${FILE_IDX}.root"
CFG_FILE="step3_${DATASET_NAME}_${FILE_IDX}_cfg.py"

# HLT_OUTPUT_FILE="output_${DATASET_NAME}_${FILE_IDX}.root"
# HLT_CFG_FILE="hltTrackingNtuple_${DATASET_NAME}_${FILE_IDX}_cfg.py"

echo ""
echo "Running cmsDriver.py..."
echo "Output file: $OUTPUT_FILE"
echo ""

cmsDriver.py step3  \
    --conditions auto:phase2_realistic_T35 \
    --datatier DQMIO \
    --eventcontent DQM \
    --geometry ExtendedRun4D110 \
    --era Phase2C17I13M9  \
    -n 100  \
    --filein  root://cmsxrootd.fnal.gov/${INPUT_FILE}  \
    --fileout file:step3.root  \
    --python_filename ${CFG_FILE} \
    --lazy_download \
    --nThreads 1 \
    --customise Validation/RecoTrack/customiseTrackingNtuple.customiseTrackingNtuple \
    --customise_command="process.trackingNtuple.clusterMasks = []" \
    -s RAW2DIGI,RECO:reconstruction_trackingOnly,VALIDATION:@trackingOnlyValidation,DQM:@trackingOnlyDQM \
    --pileup AVE_200_BX_25ns \
    --pileup_input das:/MinBias_TuneCP5_14TeV-pythia8/Phase2Spring24GS-140X_mcRun4_realistic_v4-v1/GEN-SIM \
    --procModifiers trackingLST,seedingLST,trackingMkFitHighPtTripletStep,trackingIters01 \
    --customise_command="process.trackingNtuple.seedAlgoDetect = cms.untracked.bool(False); process.trackingNtuple.seedAlgos = cms.untracked.vuint32(4)" \
    --no_exec

# no PU version
cmsDriver.py step3  \
    --conditions auto:phase2_realistic_T35 \
    --datatier DQMIO \
    --eventcontent DQM \
    --geometry ExtendedRun4D110 \
    --era Phase2C17I13M9  \
    -n 100  \
    --filein  root://cmsxrootd.fnal.gov/${INPUT_FILE}  \
    --fileout file:step3.root  \
    --python_filename ${CFG_FILE} \
    --lazy_download \
    --nThreads 1 \
    --customise Validation/RecoTrack/customiseTrackingNtuple.customiseTrackingNtuple \
    --customise_command="process.trackingNtuple.clusterMasks = []" \
    -s RAW2DIGI,RECO:reconstruction_trackingOnly,VALIDATION:@trackingOnlyValidation,DQM:@trackingOnlyDQM \
    --procModifiers trackingLST,seedingLST,trackingMkFitHighPtTripletStep,trackingIters01 \
    --customise_command="process.trackingNtuple.seedAlgoDetect = cms.untracked.bool(False); process.trackingNtuple.seedAlgos = cms.untracked.vuint32(4)" \
    --no_exec

sed -i "s/process = cms.Process('RECO'/process = cms.Process('RECOX'/" "${CFG_FILE}"
sed -i 's/numberOfStreams = cms.untracked.uint32(0)/numberOfStreams = cms.untracked.uint32(4)/' "${CFG_FILE}"
sed -i 's/numberOfThreads = cms.untracked.uint32(1)/numberOfThreads = cms.untracked.uint32(4)/' "${CFG_FILE}"

echo ""
echo "Running cmsRun..."
cmsRun ${CFG_FILE} 2>&1 | tee cmsRun.log

if [ -f "trackingNtuple.root" ]; then
    echo ""
    echo "Job completed successfully!"
    export GFAL_PYTHONBIN="/usr/bin/python3"
 
    OUTPUT_DIR=root://redirector.t2.ucsd.edu:1095//store/user/jachisma/trackingNtuples_offline_17_0_0_pre2/${DATASET_NAME}/
    gfal-copy trackingNtuple.root $OUTPUT_DIR/${OUTPUT_FILE}
    
    echo "Output copied to: $OUTPUT_DIR/${OUTPUT_FILE}"
else
    echo ""
    echo "ERROR: Output file not created!"
    exit 1
fi

cd /tmp
rm -rf $WORK_DIR
echo "Cleaned up temporary directory"
echo "=================================================="
echo "Job finished: $(date)"
