# NeurIPS 2026 Resubmission — Execution Plan (v2)

**Created**: 2026-04-21
**Target**: NeurIPS 2026 resubmission after ICML reject (4/2/3/3)
**Driver**: Stanford AI Review (post-revision) — Reviewer A lean-accept + Reviewer B weak-reject
**Decision**: Option B adopted for Fix #1 (extend SP1 circuit to enforce embedding relation on opened subset)

---

## 1. Scope: Tier 1 + Tier 2 (8 items)

| # | Subject | Type | Blocker | Est |
|---|---------|------|---------|-----|
| Prep | LaTeX broken-ref audit | sub-agent | — | 30min |
| #1-B | SP1 circuit extension (opened 32 embedding relation) | Rust code | — | 3-5 days |
| #2 | Theorem 3.5 formalization (partial opening + composite predicate) | Paper | #1-B | 1 day |
| #3 | Key hierarchy clarification (k vs sk_sig) | Paper | — | 2h |
| #4 | Proof/receipt byte-size table | Experiment+Paper | #1-B | 0.5 day |
| #5 | Canonicalization / robust-hash discussion | Paper | — | 3h |
| #6 | Key lifecycle / rotation / revocation | Paper | — | 3h |
| #7 | C2PA comparison paragraph | Paper | — | 2h |

**Tier 3 items (#8 sample expand, #9 ISTS, #10 inversion ablation) — deferred to camera-ready.**

---

## 2. Critical Path

```
Prep (latex-check) ─┐
                    │
#3, #5, #6, #7 ─────┤ (paper-only, independent)
                    │
#1-B (SP1 code) ──┬─► #2 (Theorem) ──┬──► Integration
                  └─► #4 (Proof size)┘
```

Critical path: **#1-B → #2/#4 → Integration** (~5-6 days)

---

## 3. Parallelization

### Thread A — SP1 Rust (critical path)
- Day 1–3: Circuit source ID + extension + unit tests
- Day 3–4: Proof regeneration + verification + size measurement

### Thread B — Paper writing (parallel)
- Day 1 AM: #3 (2h) + Prep (30min via sub-agent)
- Day 1 PM: #7 (2h) + #5 (3h)
- Day 2 AM: #6 (3h)
- Day 2 PM+: buffer / crop exp followup
- Day 4–5: #2 (depends on A), #4 table

### Thread C — Background
- SDXL crop experiment (GPU 0/1, ~30h remaining from 2026-04-21 08:56Z)
- 50-min auto monitoring loop active

---

## 4. Why This Order

- **Critical path first**: #1-B has longest lead time & sole engineering risk → start Day 1.
- **Parallel paper work**: #3/#5/#6/#7 independent of #1-B → knock out during Thread A runtime.
- **Thm 3.5 after circuit spec confirmed**: #2 proof wording depends on what circuit actually enforces.
- **Risk-managed**: If SP1 extension blocks (Rust issues), fallback to Option A by Day 3.

---

## 5. Risk Mitigation

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| SP1 Rust bugs | Med | Start Day 1, buffer Day 3 for debug |
| Proof regen breaks existing receipts | Low | Backup original proofs; version tag receipts |
| Theorem formalization reveals gap | Low | Option A fallback prepared |
| Crop experiment fails | Low | ~30h autonomous, already pre-flight checked |

---

## 6. Day-1 Kickoff Checklist

- [ ] Identify SP1 prover source location (`src/sp1_prover/` or equivalent)
- [ ] Locate existing circuit constraint Rust file
- [ ] Identify LUT (F⁻¹ table) implementation
- [ ] Launch latex-checker sub-agent (Prep)
- [ ] Draft Fix #3 Key Hierarchy (paper edits to §2.2 Threat Model + Table 3 caption)
