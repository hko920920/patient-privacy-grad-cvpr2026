# Existing-LoRA transfer diagnostic

This package executes the prospectively frozen 2026-09-17 comparison. It does not reopen the closed full64 branch and has no DP, expert-final, or reserved-confirmation phase.

Use the repository virtual environment, from `code_working`:

```powershell
.\base_gate\.venv\Scripts\python.exe -B -X utf8 -m lora_transfer.run prepare
.\base_gate\.venv\Scripts\python.exe -B -X utf8 -m lora_transfer.run e4_replay
.\base_gate\.venv\Scripts\python.exe -B -X utf8 -m lora_transfer.run cache_public
.\base_gate\.venv\Scripts\python.exe -B -X utf8 -m lora_transfer.run train_public
.\base_gate\.venv\Scripts\python.exe -B -X utf8 -m lora_transfer.run cache_private
.\base_gate\.venv\Scripts\python.exe -B -X utf8 -m lora_transfer.run train_pooled
.\base_gate\.venv\Scripts\python.exe -B -X utf8 -m lora_transfer.run generate
.\base_gate\.venv\Scripts\python.exe -B -X utf8 -m lora_transfer.run classifier_replay
.\base_gate\.venv\Scripts\python.exe -B -X utf8 -m lora_transfer.run classify
.\base_gate\.venv\Scripts\python.exe -B -X utf8 -m lora_transfer.run evaluate
.\base_gate\.venv\Scripts\python.exe -B -X utf8 -m lora_transfer.run verify
```

The output is `_reports/lora_transfer_20260917_v1`. `prepare` rejects an existing output. Completed phases are reused only after their SHA seals pass. Partial training runs are preserved and require an explicit technical recovery record; they are never silently overwritten or restarted. The fixed step-224 snapshots are diagnostic/recovery artifacts, and only step448 is used for generation.

The runtime contract binds the prior plan, all imported scientific sources, original controls and the new implementation. Both models continue the same learned E4 adapter independently with fresh AdamW and GradScaler states. Optimizer post-hooks and AdamW step counters distinguish successful updates from AMP skips. The public-only process cannot open the private cache; the broad historical mixed cache is forbidden.

The classifier uses unchanged `downstream_utility.train_v2` and `data_v2`. Its temporary batch-provider injection is restored in `finally`. Six extra one-step updates check old-kernel equality and both actual new sources, separately from the six 400-step runs. Two extra E4 decodes check full-trajectory equality. Legacy `data`/`train` imports are blocked.

Primary utility is the pooled-LoRA difference over public-LoRA and both real-only controls on the existing weak-label method-development cohort. A technical integrity PASS is separate from the engineering utility gate. One LoRA trajectory and one bank per arm with three classifier seeds do not establish generator-training replication or independent confirmation.
