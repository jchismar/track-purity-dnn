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
CMSSW_VERSION=CMSSW_16_0_0_pre3

cd /ceph/users/aaarora/${CMSSW_VERSION}/src/
cmsenv

cd $CURRENT_DIR

WORK_DIR=$(mktemp -d)
cd $WORK_DIR
echo "Working in temporary directory: $WORK_DIR"

OUTPUT_FILE="output_Phase2_L1T_${DATASET_NAME}_${FILE_IDX}.root"
CFG_FILE="rerunL1_${DATASET_NAME}_${FILE_IDX}_cfg.py"

HLT_OUTPUT_FILE="output_${DATASET_NAME}_${FILE_IDX}.root"
HLT_CFG_FILE="hltTrackingNtuple_${DATASET_NAME}_${FILE_IDX}_cfg.py"

echo ""
echo "Running cmsDriver.py..."
echo "Output file: $OUTPUT_FILE"
echo ""

cmsDriver.py -s L1TrackTrigger,L1 \
    --conditions auto:phase2_realistic_T33 \
    --geometry ExtendedRun4D110 \
    --era Phase2C17I13M9 \
    --eventcontent FEVTDEBUGHLT \
    --datatier GEN-SIM-DIGI-RAW-MINIAOD \
    --customise SLHCUpgradeSimulations/Configuration/aging.customise_aging_1000,Configuration/DataProcessing/Utils.addMonitoring,L1Trigger/Configuration/customisePhase2TTOn110.customisePhase2TTOn110 \
    --filein root://cmsxrootd.fnal.gov/${INPUT_FILE} \
    --fileout "file:${OUTPUT_FILE}" \
    --python_filename "${CFG_FILE}" \
    --inputCommands="keep *, drop l1tPFJets_*_*_*, drop l1tTrackerMuons_l1tTkMuonsGmt*_*_HLT" \
    --mc -n 100 \
    --no_exec

sed -i 's/numberOfStreams = cms.untracked.uint32(0)/numberOfStreams = cms.untracked.uint32(4)/' "${CFG_FILE}"
sed -i 's/numberOfThreads = cms.untracked.uint32(1)/numberOfThreads = cms.untracked.uint32(4)/' "${CFG_FILE}"

echo ""
echo "Running cmsRun..."
cmsRun ${CFG_FILE} 2>&1 | tee cmsRun.log

cmsDriver.py  \
    step2 -s L1P2GT,HLT:75e33_trackingOnly,VALIDATION:@hltValidation \
    --conditions auto:phase2_realistic_T33 \
    --datatier DQMIO \
    --eventcontent DQMIO \
    --procModifiers phase2CAExtension,singleIterPatatrack,trackingLST,seedingLST,trackingMkFitCommon,hltTrackingMkFitInitialStep \
    --geometry ExtendedRun4D110 \
    --era Phase2C17I13M9 \
    --filein file:${OUTPUT_FILE} \
    --fileout file:output.root \
    --python_filename ${HLT_CFG_FILE} \
    --processName=HLTX \
    --inputCommands='keep *, drop *_hlt*_*_HLT, drop triggerTriggerFilterObjectWithRefs_l1t*_*_HLT' \
    --outputCommands='drop *_*_*_HLT' \
    --customise Validation/RecoTrack/customiseTrackingNtuple.customiseTrackingNtupleHLT,Validation/RecoTrack/customiseTrackingNtuple.extendedContent \
    --accelerators cpu \
    --no_exec --mc -n -1

sed -i 's/numberOfStreams = cms.untracked.uint32(0)/numberOfStreams = cms.untracked.uint32(4)/' "${HLT_CFG_FILE}"
sed -i 's/numberOfThreads = cms.untracked.uint32(1)/numberOfThreads = cms.untracked.uint32(4)/' "${HLT_CFG_FILE}"

cmsRun ${HLT_CFG_FILE} 2>&1 | tee -a cmsRun.log

if [ -f "$OUTPUT_FILE" ]; then
    echo ""
    echo "Job completed successfully!"
    echo "Output file size: $(du -h $OUTPUT_FILE)"
    export GFAL_PYTHONBIN="/usr/bin/python3"
 
    OUTPUT_DIR=root://redirector.t2.ucsd.edu:1095//store/user/aaarora/tracking/
    gfal-copy trackingNtuple.root $OUTPUT_DIR/${HLT_OUTPUT_FILE}
    
    echo "Output copied to: $OUTPUT_DIR/${HLT_OUTPUT_FILE}"
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
