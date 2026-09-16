"""Check runtime report arithmetic, sources, links, and actual Chrome calculator changes."""
from pathlib import Path
import json, hashlib, math, subprocess, tempfile, time, shutil
import requests, websocket
import verify_review

ROOT = Path(__file__).resolve().parent

def browser_check():
    profile = Path(tempfile.mkdtemp(prefix="runtime-ui-check-", dir=ROOT)).resolve()
    proc = None
    ws = None
    try:
        proc = subprocess.Popen([
            "C:/Program Files/Google/Chrome/Application/chrome.exe",
            "--headless=new", "--no-first-run", "--no-default-browser-check", "--disable-gpu",
            "--remote-debugging-port=0", "--remote-allow-origins=*",
            "--user-data-dir=" + str(profile), "about:blank"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
        portfile = profile / "DevToolsActivePort"
        for _ in range(120):
            if portfile.exists(): break
            time.sleep(.1)
        port = int(portfile.read_text().splitlines()[0])
        targets = requests.get(f"http://127.0.0.1:{port}/json", timeout=5).json()
        page = next(t for t in targets if t["type"] == "page")
        ws = websocket.create_connection(page["webSocketDebuggerUrl"], timeout=10, suppress_origin=True)
        seq = 0
        errors = []
        def call(method, params=None):
            nonlocal seq
            seq += 1
            request_id = seq
            ws.send(json.dumps({"id": request_id, "method": method, "params": params or {}}))
            while True:
                m = json.loads(ws.recv())
                if m.get("method") == "Runtime.exceptionThrown": errors.append(m)
                if m.get("id") == request_id:
                    assert "error" not in m, m
                    return m.get("result", {})
        def evaluate(js):
            result = call("Runtime.evaluate", {"expression": js, "returnByValue": True})
            assert "exceptionDetails" not in result, result
            return result["result"].get("value")
        call("Runtime.enable")
        call("Page.enable")
        call("Page.navigate", {"url": (ROOT / "runtime_review.html").as_uri()})
        for _ in range(100):
            if evaluate("document.readyState==='complete' && typeof updateBudget==='function'"): break
            time.sleep(.1)
        first = evaluate("document.getElementById('budget-output').textContent")
        assert "25.6" in first, first
        changed = evaluate("""
            document.getElementById('nodes').value='2';
            document.getElementById('daily').value='16';
            document.getElementById('util').value='0.6';
            document.getElementById('scenario').value='2';
            document.getElementById('scenario').dispatchEvent(new Event('change'));
            document.getElementById('budget-output').textContent
        """)
        assert "214.1" in changed, changed
        subset = evaluate("""
            document.getElementById('extension').value='0';
            document.getElementById('extension').dispatchEvent(new Event('change'));
            document.getElementById('budget-output').textContent
        """)
        assert "143.6" in subset, subset
        assert not errors, errors
        result = dict(status="PASS",default=first,changed=changed,m2_only=subset,javascript_exceptions=len(errors))
        try: call("Browser.close")
        except Exception: pass
        return result
    finally:
        if ws:
            try: ws.close()
            except Exception: pass
        if proc:
            try: proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.terminate()
                proc.wait(timeout=5)
        assert profile.parent == ROOT.resolve() and profile.name.startswith("runtime-ui-check-")
        for _ in range(20):
            try:
                shutil.rmtree(profile)
                break
            except PermissionError: time.sleep(.2)

def main():
    d = json.loads((ROOT / "runtime_budget.json").read_text(encoding="utf-8"))
    assert d["experiment_executed"] is False
    assert len(d["methods"]) == 8
    assert d["cohort_metadata"]["members"]["E_m2"] == d["cohort_metadata"]["nonmembers"]["E_m2"] == 865
    assert d["cohort_metadata"]["members"]["E_m5"] == d["cohort_metadata"]["nonmembers"]["E_m5"] == 304
    assert math.isclose(1730 * 12 * 30 / 3600, 173)
    assert math.isclose(608 * 12 * 75 / 3600, 152)
    assert math.isclose(1730 * 2 * 12 * 7 / 60, 4844)
    for i in range(3):
        assert math.isclose(sum(m["hours_E2"][i] for m in d["methods"]), next(s["hours"][i] for s in d["stages"] if s["id"] == "D4_E2"))
        assert math.isclose(sum(s["hours"][i] for s in d["stages"]), d["spark_device_hours_core"][i])
        assert math.isclose(d["spark_device_hours_core"][i] + d["extension"]["hours"][i], d["spark_device_hours_with_m5"][i])
    for rel, digest in d["source_sha256"].items():
        assert hashlib.sha256((ROOT.parent.parent / rel).read_bytes()).hexdigest() == digest, rel
    report = dict(arithmetic="PASS",sources_checked=len(d["source_sha256"]),
                  static=verify_review.static_check(),browser=browser_check(),
                  training_or_attack_executed=False)
    (ROOT/"runtime_verification.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False))

if __name__=="__main__":
    main()

