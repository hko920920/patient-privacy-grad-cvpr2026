"""Re-execute frozen verification on CPU, redirecting reports to a new audit directory."""
import contextlib, importlib, io, json, sys, time, unittest
from pathlib import Path
from .common import ROOT, RUN, digest, write_json

OUT=RUN/"verification_20260914/foundation"

def main():
    if (OUT/"report.json").exists():raise RuntimeError("completed foundation verification exists")
    OUT.mkdir(parents=True,exist_ok=True)
    protected=[RUN/"cohort/lock.json",RUN/"cache/summary.json",
               RUN/"training_coverage_v2/protocol.json"]
    protected += [RUN/"training_coverage_v2"/m/"step_1000.pt" for m in ["model_1","model_2"]]
    protected += list((RUN/"probe_smoke/training_coverage_v2").glob("*/results.json"))
    protected += [Path(__file__).with_name(n) for n in ["probe.py","models.py","train_coverage.py","train_pair.py"]]
    before={str(p.relative_to(ROOT)):digest(p) for p in protected}
    protocol=json.loads((RUN/"training_coverage_v2/protocol.json").read_text())
    for name,key in [("train_coverage.py","code_sha256"),("models.py","models_code_sha256"),("train_pair.py","save_helper_code_sha256")]:
        assert digest(Path(__file__).with_name(name))==protocol[key]
    cases=[("verify_cohort",[]),("verify_training",["--model","model_1","--step","1000"]),
           ("verify_training",["--model","model_2","--step","1000"]),
           ("verify_quality",[]),("verify_probe",["--scenario","U","--count","8"]),
           ("verify_probe",["--scenario","E","--count","1"]),("verify_u_analysis",[])]
    records=[]
    for name,args in cases:
        module=importlib.import_module("u_patient_audit."+name)
        emitted=[]
        def redirect(path,value):
            path=Path(path)
            destination=OUT/"rechecked"/path.relative_to(RUN)
            write_json(destination,value);emitted.append(dict(path=str(destination.relative_to(RUN)),status=value.get("status")))
        original=module.write_json;module.write_json=redirect
        old_argv=sys.argv;sys.argv=[name]+args
        capture=io.StringIO();start=time.perf_counter()
        try:
            with contextlib.redirect_stdout(capture):module.main()
        finally:
            sys.argv=old_argv;module.write_json=original
        entry=dict(check=name,args=args,status="PASS",seconds=time.perf_counter()-start,outputs=emitted)
        records.append(entry);print(json.dumps(entry),flush=True)
    tests=importlib.import_module("u_patient_audit.test_core")
    stream=io.StringIO();start=time.perf_counter()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
    (OUT/"unit_tests.txt").write_text(stream.getvalue(),encoding="utf-8")
    assert result.wasSuccessful()
    after={str(p.relative_to(ROOT)):digest(p) for p in protected}
    assert before==after
    report=dict(status="PASS_FROZEN_FOUNDATION_RECHECK",checks=records,unit_tests=result.testsRun,
                unit_seconds=time.perf_counter()-start,protected_files_unchanged=len(before),protected_sha256=before,
                original_reports_overwritten=False,new_GPU_model_execution=False,
                scope="recheck existing evidence and CPU tests; not nonzero gradient or attack efficacy verification")
    write_json(OUT/"report.json",report)
    print(json.dumps(report))
if __name__=="__main__":main()

