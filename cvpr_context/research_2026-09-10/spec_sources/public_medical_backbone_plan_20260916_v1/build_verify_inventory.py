"""Metadata/file-byte inventory only; never loads a model or decodes an image."""
from pathlib import Path
from collections import Counter, defaultdict
import argparse, csv, hashlib, io, json, time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CODE = ROOT / "code_working"
SALT = "public-medical-backbone-20260916-v1"

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def read(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def rel(path):
    return Path(path).relative_to(ROOT).as_posix()

def encoded_csv(rows):
    s = io.StringIO(newline="")
    w = csv.DictWriter(s, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return s.getvalue().encode("utf-8")

def key(pid):
    return hashlib.sha256((SALT + "|patient|" + pid).encode("utf-8")).hexdigest()

def assemble():
    paths = {
        "source": CODE / "_data/derived/nih_cxr14_pa_target_enriched_v1/all_selected_pa_images_private.csv",
        "patients": CODE / "_data/derived/nih_cxr14_pa_target_enriched_v1/patients_private.csv",
        "k5": CODE / "_data/derived/nih_cxr14_pa_target_enriched_v1/k5_private.csv",
        "inventory": CODE / "_data/raw/nih_cxr14_pa_k10_plus_census_v1/content_inventory_private.csv",
        "evaluation": CODE / "_reports/cvpr_u_pilot_v1_001/cohort/evaluation_images.csv",
        "auxiliary": CODE / "_reports/cvpr_u_pilot_v1_001/cohort/auxiliary_images.csv",
        "source_audit": CODE / "_reports/cvpr_u_pilot_v1_001/cohort/source_audit.csv",
        "old_reference448": CODE / "_reports/nih_cxr14_k5_evaluator_preflight_protocol_v1_001/real_evaluator_tasks_local.csv",
        "triplets": HERE.parent / "protection_observation_design_20260915/public_triplet_fixture.json",
        "fixture_source": CODE / "dp_training/run_k5_actual_sd21_resume_preflight.py",
        "fixture_protocol": CODE / "_reports/nih_cxr14_k5_actual_sd21_resume_preflight_protocol_v1_002/protocol.json",
    }
    source = read(paths["source"])
    pats = {r["patient_id"]: r for r in read(paths["patients"])}
    inv = {r["image_id"]: r for r in read(paths["inventory"])}
    audited = {r["image_id"]: r for r in read(paths["source_audit"])}
    ev, aux, refs = (read(paths[k]) for k in ["evaluation", "auxiliary", "old_reference448"])
    pub = [r for r in source if r["partition"] == "public_development"]
    pubids = {r["patient_id"] for r in pub}
    roles = {x: {r["patient_id"] for r in ev if r["eval_role"] == x}
             for x in ["fit", "selection", "calibration", "test"]}
    roles.update({x: {r["patient_id"] for r in aux if r["eval_role"] == x}
                  for x in ["quality", "reference", "background"]})
    protected = set.union(*(roles[x] for x in ["fit", "selection", "calibration", "test", "quality"]))
    roles["reference259"] = {r["patient_id"] for r in refs} - protected
    assert len(roles["reference259"]) == 259
    excluded = set.union(*roles.values())
    eligible = pubids - excluded
    assert len(eligible) == 905

    histories = defaultdict(set)
    history_inputs = sorted(p for d in (CODE / "_reports").glob("nih_cxr14*")
                            if d.is_dir() for p in d.glob("*.csv")
                            if p.name.startswith("selected_"))
    for p in history_inputs:
        paths["history:" + rel(p)] = p
        for row in read(p):
            for k, v in row.items():
                if (k == "patient_id" or k.endswith("_patient_id")) and v in eligible:
                    histories[v].add(rel(p))
    trip = json.loads(paths["triplets"].read_text(encoding="utf-8"))
    for row in trip["records"]:
        if row["patient_id"] in eligible:
            histories[row["patient_id"]].add(rel(paths["triplets"]))
    # Exact published preflight selection: lexical patient IDs, >=2 K5 records.
    k5 = [r for r in read(paths["k5"]) if r["partition"] == "public_development"]
    counts = Counter(r["patient_id"] for r in k5)
    fixture = sorted(p for p, n in counts.items() if n >= 2)[:8]
    for p in set(fixture) & eligible:
        histories[p].add("K5_four_step_fixture:" + rel(paths["fixture_protocol"]))
    historical = {p for p in eligible if histories[p]}
    no_known_history = eligible - historical
    assert len(no_known_history) >= 256
    order = sorted(eligible, key=key)
    clean_order = sorted(no_known_history, key=key)
    confirmation = set(clean_order[:128])
    selection = set(clean_order[128:256])
    rest = sorted(eligible - confirmation - selection, key=key)
    train, reserve = set(rest[:640]), set(rest[640:])
    assert len(train) == 640 and len(reserve) == 9
    assign = {p: name for name, group in [
        ("train", train), ("selection", selection),
        ("confirmation", confirmation), ("reserve", reserve)] for p in group}
    assert len(assign) == len(eligible)
    assert not (selection | confirmation) & historical

    image_root = CODE / "_data/raw/nih_cxr14_pa_k10_plus_census_v1/images"
    rows = [r for r in pub if r["patient_id"] in eligible]
    assert len(rows) == 1016
    image_counts = Counter(r["patient_id"] for r in rows)
    bytes_to_patients = defaultdict(set)
    for r in inv.values():
        bytes_to_patients[r["sha256"].lower()].add(r["patient_id"])
    pixels_to_patients = defaultdict(set)
    for r in audited.values():
        pixels_to_patients[r["pixel_sha256"]].add(r["patient_id"])
    image_rows = []
    for r in sorted(rows, key=lambda x: (key(x["patient_id"]), x["image_id"])):
        p, iid = r["patient_id"], r["image_id"]
        f = image_root / r["image_filename"]
        assert f.is_file() and iid in inv and iid in audited
        h = sha(f)
        assert h == inv[iid]["sha256"].lower() == audited[iid]["sha256"].lower()
        assert bytes_to_patients[h] == {p}, ("cross-patient byte duplicate", iid)
        assert pixels_to_patients[audited[iid]["pixel_sha256"]] == {p}
        assert r["view"] == "PA" and r["official_source_split"] == "train_val"
        image_rows.append(dict(r, backbone_role=assign[p], ordering_sha256=key(p),
                               image_path=rel(f), sha256=h, bytes=f.stat().st_size,
                               prior_pixel_sha256=audited[iid]["pixel_sha256"],
                               known_diagnostic_history=int(bool(histories[p]))))
    patient_rows = [
        {"patient_id": p, "source_partition": pats[p]["partition"],
         "backbone_role": assign[p], "ordering_sha256": key(p),
         "available_images": image_counts[p], "target_patient": pats[p]["target_patient"],
         "known_diagnostic_history": int(bool(histories[p])),
         "history_sources": "|".join(sorted(histories[p]))}
        for p in order]
    history_rows = [{"patient_id": p, "backbone_role": assign[p], "source": s}
                    for p in order for s in sorted(histories[p])]
    old_assign = {p: ("train" if i < 640 else "selection" if i < 768 else
                      "confirmation" if i < 896 else "reserve") for i, p in enumerate(order)}
    inputs = {rel(p): sha(p) for p in sorted(set(paths.values()))}
    details = {
        "schema": "public-medical-backbone-role-plan/v1",
        "status": "METADATA_AND_FILE_BYTES_VERIFIED_NO_TRAINING",
        "salt": SALT, "ordering_rule": "ascending SHA256(UTF8(salt+'|patient|'+patient_id)); numeric ID text as in original manifest",
        "assignment_rule": "known-history0 confirmation128 first; next known-history0 selection128; remaining all train640 then reserve9 in same SHA order",
        "public_role_assumption": "Existing public_development is available for public backbone training. These patients themselves receive no privacy guarantee. Original roles elsewhere are unchanged.",
        "source_inventory": {"local_image_files": len(list(image_root.glob("*.png"))),
            "inventory_images": len(inv), "inventory_patients": len({r["patient_id"] for r in inv.values()}),
            "source_public_patients": len(pubids), "source_public_manifest_images": len(pub),
            "source_public_acquired_images": sum(r["image_filename"] in inv for r in pub)},
        "excluded_patient_counts": {k: len(v) for k, v in roles.items()},
        "excluded_union": len(excluded),
        "eligible_patients": len(eligible), "eligible_images": len(rows),
        "patients_by_role": dict(Counter(assign.values())),
        "images_by_role": dict(Counter(r["backbone_role"] for r in image_rows)),
        "images_per_patient_histogram": dict(sorted(Counter(image_counts.values()).items())),
        "target_patient_counts_by_role": {
            name: dict(Counter(pats[p]["target_patient"] for p in group))
            for name, group in [("train", train), ("selection", selection), ("confirmation", confirmation), ("reserve", reserve)]},
        "historical_diagnostic_patients": len(historical),
        "no_known_history_patients": len(no_known_history),
        "history_patients_by_role": dict(Counter(assign[p] for p in historical)),
        "history_sources_patients": {s: sum(s in histories[p] for p in eligible)
                                    for s in sorted(set().union(*histories.values()))},
        "history_change_record": {
            "before": "Unexecuted calculation only: first640 train,next128 selection,next128 confirmation,last9 reserve.",
            "initial_rule_images_by_role": dict(Counter(old_assign[r["patient_id"]] for r in rows)),
            "initial_rule_history_by_role": dict(Counter(old_assign[p] for p in historical)),
            "reason": "Before training or outcome inspection, reserve selection/confirmation without overlap in inspected diagnostic manifests.",
            "previous_manifest_published": False},
        "file_checks": {"actual_image_sha256_checked": len(rows), "all_match_content_inventory_and_source_audit": True,
                        "cross_patient_exact_byte_matches": 0, "cross_patient_prior_pixel_hash_matches": 0,
                        "new_png_decodes": 0, "near_duplicate_screen_new": False},
        "cache_boundary": "All original CVPR evaluation/auxiliary patients excluded; the existing 2304-image cache cannot supply these newly reserved images. New latent/text preparation is required before training.",
        "history_scope": "Inspected selected_*.csv in NIH report directories (including negative/donor patient IDs), public-triplet fixture, and source-defined K5 four-step fixture. All public images also had acquisition/source-audit preprocessing. Zero known history is not proof of never being viewed, nor an independent clinical/external dataset.",
        "input_sha256": inputs, "builder_sha256": sha(__file__),
        "limitations": ["Patient identities/splits are NIH metadata, not clinical study UID.",
            "Same NIH PA population, not an external dataset.",
            "Historical reference259 includes original auxiliary patients; exclusion is a union, not subtraction of summed counts.",
            "No feature/loss/generation score used for role assignment.",
            "No training, backbone evaluation, private reclassification, or final calibration/test image access performed.",
            "Remaining 905 patients are mostly single-image and have a different case mix from prior >=4-image CVPR cohort.",
            "No new perceptual near-duplicate search was run; exact byte and previously audited pixel hashes were checked."] }
    return patient_rows, image_rows, history_rows, details

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    patients, images, histories, details = assemble()
    blobs = {"patients.csv": encoded_csv(patients), "images.csv": encoded_csv(images),
             "diagnostic_history.csv": encoded_csv(histories)}
    details["output_sha256"] = {n: hashlib.sha256(b).hexdigest() for n, b in blobs.items()}
    blobs["exclusion_calculation.json"] = (json.dumps(details, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if args.verify:
        for n, b in blobs.items():
            assert (HERE / n).read_bytes() == b, "Saved manifest mismatch: " + n
    else:
        for n, b in blobs.items():
            with (HERE / n).open("xb") as f:
                f.write(b)
    result = {"status": "PASS_MANIFEST_RECONSTRUCTION_AND_1016_FILE_HASHES" if args.verify else "PLAN_MANIFEST_WRITTEN",
              "patients": len(patients), "images": len(images),
              "patients_by_role": details["patients_by_role"], "images_by_role": details["images_by_role"],
              "historical_diagnostic_patients": details["historical_diagnostic_patients"],
              "history_patients_by_role": details["history_patients_by_role"],
              "seconds": time.perf_counter() - started,
              "output_sha256": {n: sha(HERE / n) for n in blobs},
              "code_sha256": sha(__file__)}
    if args.verify:
        with (HERE / "verification.json").open("x", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            f.write("\n")
    print(json.dumps(result, ensure_ascii=False))

if __name__ == "__main__":
    main()

