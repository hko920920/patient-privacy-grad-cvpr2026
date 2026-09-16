#!/usr/bin/env python3
"""Attack harness scaffolding for PP-Mark v0.3."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageChops


@dataclass(slots=True)
class AttackSpec:
    name: str
    description: str
    requires_gpu: bool = True
    requires_external_tool: bool = False


@dataclass(slots=True)
class AttackResult:
    name: str
    status: str
    reason: str


ATTACKS: List[AttackSpec] = [
    AttackSpec(name="mueller_forgery", description="Müller semantic forgery attack."),
    AttackSpec(name="mueller_erasure", description="Müller erasure attack."),
    AttackSpec(name="mueller_reprompt", description="Müller re-prompt attack."),
    AttackSpec(name="zhao_regen", description="Zhao diffusion re-generation attack."),
    AttackSpec(name="geometry_suite", description="Rotation/scale/crop sweep", requires_gpu=False),
    AttackSpec(name="image_quality", description="LPIPS/SSIM quality report", requires_external_tool=True),
    AttackSpec(name="zkp_performance", description="Halo2 proving/verifying latency", requires_gpu=False),
]


def execute_attack(spec: AttackSpec) -> AttackResult:
    reason_parts: List[str] = []
    if spec.requires_gpu:
        reason_parts.append("requires GPU environment")
    if spec.requires_external_tool:
        reason_parts.append("requires external toolchain")
    reason = ", ".join(reason_parts) or "pending implementation"
    return AttackResult(name=spec.name, status="skipped", reason=reason)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PP-Mark v0.3 attack suite (scaffold)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image", help="Watermarked image path for image-based attacks")
    parser.add_argument("--max-shift", type=int, default=4)
    parser.add_argument("--max-rotation", type=float, default=2.0)
    parser.add_argument("--rotation-step", type=float, default=1.0)
    parser.add_argument("--sync-mode", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--regen-cmd-template", help="Command template for Zhao regen (with {prompt} {output})")
    parser.add_argument("--prompts-file", help="Prompts file for Zhao regen")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with Path(args.metadata).open("r", encoding="utf-8") as handle:
        meta = json.load(handle)

    def run_verifier(img_path: Path) -> bool:
        cmd = [
            "python",
            "-m",
            "ppmark_v03.cli",
            "verifier",
            "--config",
            args.config,
            "--metadata",
            args.metadata,
            "--image",
            str(img_path),
            "--max-shift",
            str(args.max_shift),
            "--max-rotation",
            str(args.max_rotation),
            "--rotation-step",
            str(args.rotation_step),
            "--sync-mode",
            args.sync_mode,
        ]
        try:
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError:
            return False

    def add_result(name: str, status: str, reason: str) -> None:
        results.append(AttackResult(name=name, status=status, reason=reason))

    results: List[AttackResult] = []
    tmp_dir = Path(tempfile.mkdtemp(prefix="attack_suite_"))
    base_img: Image.Image | None = None
    if args.image:
        base_img = Image.open(args.image).convert("RGB")

    for spec in ATTACKS:
        if spec.name == "zkp_performance":
            halo2_meta = meta.get("halo2", {})
            timings = meta.get("timings", {})
            proof_path = Path(halo2_meta.get("proof", ""))
            proof_size = proof_path.stat().st_size if proof_path.exists() else 0
            prover_t = timings.get("halo2_prover_sec")
            reason = f"proof_size_bytes={proof_size}"
            if prover_t is not None:
                reason += f", prover_sec={prover_t:.3f}"
            results.append(AttackResult(name=spec.name, status="measured", reason=reason))
            continue
        if spec.name in ("mueller_forgery", "mueller_erasure", "mueller_reprompt"):
            add_result(spec.name, "skipped", "requires GPU/external attack implementation")
        elif spec.name == "zhao_regen":
            if not base_img or not args.regen_cmd_template or not args.prompts_file:
                add_result(spec.name, "skipped", "image or regen command/prompts missing")
            else:
                passes = 0
                total = 0
                prompts = [
                    p.strip()
                    for p in Path(args.prompts_file).read_text(encoding="utf-8").splitlines()
                    if p.strip()
                ]
                for idx, prompt in enumerate(prompts):
                    out_path = tmp_dir / f"zhao_regen_{idx}.png"
                    cmd_str = args.regen_cmd_template.format(prompt=prompt, output=out_path)
                    try:
                        subprocess.run(cmd_str.split(), check=True)
                        total += 1
                        if run_verifier(out_path):
                            passes += 1
                    except subprocess.CalledProcessError:
                        total += 1
                reason = f"pass={passes}/{total}"
                add_result(spec.name, "measured" if total else "skipped", reason)
        elif spec.name == "geometry_suite":
            if not base_img:
                add_result(spec.name, "skipped", "image missing")
            else:
                transforms = []
                for deg in [-2, 0, 2]:
                    transforms.append((f"rot_{deg}", base_img.rotate(deg, resample=Image.BICUBIC)))
                for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2), (0, 0)]:
                    transforms.append((f"shift_{dx}_{dy}", ImageChops.offset(base_img, dx, dy)))
                for scale in [0.98, 1.0, 1.02]:
                    w, h = base_img.size
                    target = (max(1, int(w * scale)), max(1, int(h * scale)))
                    transforms.append((f"scale_{scale}", base_img.resize(target, resample=Image.BICUBIC)))
                for q in [95, 85, 75]:
                    jpeg_path = tmp_dir / f"jpeg_{q}.jpg"
                    base_img.save(jpeg_path, format="JPEG", quality=q, subsampling=2)
                    transforms.append((f"jpeg_{q}", Image.open(jpeg_path).convert("RGB")))
                passes = 0
                total = 0
                for name, img in transforms:
                    out_path = tmp_dir / f"{name}.png"
                    img.save(out_path)
                    total += 1
                    if run_verifier(out_path):
                        passes += 1
                reason = f"pass={passes}/{total}"
                add_result(spec.name, "measured", reason)
        elif spec.name == "image_quality":
            add_result(spec.name, "skipped", "quality metrics not implemented")
        else:
            results.append(execute_attack(spec))

    payload: Dict[str, Dict[str, str]] = {
        result.name: {
            "status": result.status,
            "reason": result.reason,
        }
        for result in results
    }
    report_path = output_dir / "attack_results.json"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Attack suite report written to {report_path}")


if __name__ == "__main__":
    main()
