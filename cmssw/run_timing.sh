#!/bin/bash
set -e

cmsDriver.py Phase2 -s L1P2GT,HLT:75e33_timing --processName=HLTX \
    --conditions auto:phase2_realistic_T33 \
    --geometry ExtendedRun4D110 \
    --era Phase2C17I13M9 \
    --eventcontent FEVTDEBUGHLT \
    --customise SLHCUpgradeSimulations/Configuration/aging.customise_aging_1000 \
    --filein file:/ceph/cms/store/user/mmasciov/HLTPhase2/output_TTPU_Phase2_L1T.root \
    --inputCommands='keep *, drop *_hlt*_*_HLT, drop triggerTriggerFilterObjectWithRefs_l1t*_*_HLT' \
    --mc \
    --procModifier singleIterPatatrack,phase2CAExtension,trackingLST,seedingLST,trackingMkFitCommon,hltTrackingMkFitInitialStep \
    -n 100 --nThreads 1 --accelerators cpu --output={}

mv Phase2Timing_resources.json Phase2Timing_resources_MVA_CPU.json

cmsDriver.py Phase2 -s L1P2GT,HLT:75e33_timing --processName=HLTX \
    --conditions auto:phase2_realistic_T33 \
    --geometry ExtendedRun4D110 \
    --era Phase2C17I13M9 \
    --eventcontent FEVTDEBUGHLT \
    --customise SLHCUpgradeSimulations/Configuration/aging.customise_aging_1000 \
    --filein file:/ceph/cms/store/user/mmasciov/HLTPhase2/output_TTPU_Phase2_L1T.root \
    --inputCommands='keep *, drop *_hlt*_*_HLT, drop triggerTriggerFilterObjectWithRefs_l1t*_*_HLT' \
    --mc \
    --procModifier singleIterPatatrack,phase2CAExtension,trackingLST,seedingLST,trackingMkFitCommon,hltTrackingMkFitInitialStep \
    -n 100 --nThreads 1 --accelerators gpu-nvidia --output={}

mv Phase2Timing_resources.json Phase2Timing_resources_MVA_GPU.json