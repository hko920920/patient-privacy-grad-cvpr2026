# MoFit COCO public scalar replay

Stage 2, public stored results only. No model inference, GPU, or patient data.

- Original evaluator selection: gamma=0.55, ASR=88.30%, grid AUC=94.1948%, first-FPR>=1% TPR=46.80%.
- Independent rank AUC=94.3940%; empirical max TPR at FPR<=1%=48.80% (actual FPR=1.00%).
- Four released files each contain 500 numeric records; all input/source hashes unchanged.
- Embedding header says 1000 embedding iterations; paper/shell default is 300. Sample IDs are absent.
- Full-fixture parameter selection reproduces the published evaluator procedure; it is not held-out calibration.
- This result verifies the stored score path, not original model inference or patient-U performance.
