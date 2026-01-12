#!/bin/bash
set -e

NAME=$1

cmsDriver.py Phase2 -s L1P2GT,HLT:75e33 --processName=HLTX \
    --conditions auto:phase2_realistic_T33 \
    --geometry ExtendedRun4D110 \
    --era Phase2C17I13M9 \
    --eventcontent FEVTDEBUGHLT \
    --customise SLHCUpgradeSimulations/Configuration/aging.customise_aging_1000 \
    --filein file:/ceph/cms/store/user/mmasciov/HLTPhase2/output_TTPU_Phase2_L1T.root \
    --inputCommands='keep *, drop *_hlt*_*_HLT, drop triggerTriggerFilterObjectWithRefs_l1t*_*_HLT' \
    --mc \
    --procModifier singleIterPatatrack,phase2CAExtension,trackingLST,seedingLST,trackingMkFitCommon,hltTrackingMkFitInitialStep \
    -n -1 --nThreads 16

cmsDriver.py DQM -s VALIDATION:hltMultiTrackValidation \
    --conditions auto:phase2_realistic_T33 \
    --geometry ExtendedRun4D110 \
    --era Phase2C17I13M9 \
    --datatier DQMIO \
    --eventcontent DQM \
    --filein file:Phase2_L1P2GT_HLT.root \
    --hltProcess HLTX \
    --fileout ${NAME}_VAL.root \
    -n -1 --nThreads 16 \
    --procModifier singleIterPatatrack,phase2CAExtension,trackingLST,seedingLST,trackingMkFitCommon,hltTrackingMkFitInitialStep

cmsDriver.py HARVEST -s HARVESTING:@trackingOnlyValidation+@trackingOnlyDQM+postProcessorHLTtrackingSequence \
    --filein file:${NAME}_VAL.root \
    --scenario pp \
    --filetype DQM \
    --conditions auto:phase2_realistic_T33 \
    --mc -n -1

mv DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root ${NAME}.root