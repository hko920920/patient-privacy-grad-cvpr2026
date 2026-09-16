import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from opacus import PrivacyEngine
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from opacus_uci_har_dpsgd_demo import (
    WINDOW_LENGTH,
    WINDOW_STEP,
    base_records,
    load_split,
    non_overlap,
)
from unit_audit import canonical_records_sha256, file_sha256


SIGNAL_FILES = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
]


class TinySequenceCNN(nn.Module):
    def __init__(self, channels, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(16, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class SmallSequenceTCN(nn.Module):
    def __init__(self, channels, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=5, padding=2),
            nn.GroupNorm(4, 32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, padding=2, dilation=1),
            nn.GroupNorm(4, 32),
            nn.ReLU(),
            nn.Conv1d(32, 48, kernel_size=5, padding=4, dilation=2),
            nn.GroupNorm(6, 48),
            nn.ReLU(),
            nn.Conv1d(48, 48, kernel_size=5, padding=8, dilation=4),
            nn.GroupNorm(6, 48),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(48, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x))


def make_model(name, channels, num_classes):
    if name == "tiny_cnn":
        return TinySequenceCNN(channels=channels, num_classes=num_classes)
    if name == "small_tcn":
        return SmallSequenceTCN(channels=channels, num_classes=num_classes)
    raise ValueError(f"Unknown model: {name}")


def load_inertial_split(root, split):
    split_dir = root / split / "Inertial Signals"
    channels = []
    for name in SIGNAL_FILES:
        channels.append(np.loadtxt(split_dir / f"{name}_{split}.txt").astype(np.float32))
    x = np.stack(channels, axis=1)
    y = np.loadtxt(root / split / f"y_{split}.txt", dtype=int).astype(np.int64) - 1
    subjects = np.loadtxt(root / split / f"subject_{split}.txt", dtype=int)
    return x, y, subjects


def standardize(train_x, test_x):
    mean = train_x.mean(axis=(0, 2), keepdims=True)
    std = train_x.std(axis=(0, 2), keepdims=True)
    std = np.maximum(std, 1e-6)
    return (train_x - mean) / std, (test_x - mean) / std


def subset_x_y(x, y, records):
    idx = [int(r["row_index"]) for r in records]
    return x[idx], y[idx]


def train_private_cnn(
    train_x,
    train_y,
    test_x,
    test_y,
    target_epsilon,
    target_delta,
    epochs,
    batch_size,
    learning_rate,
    max_grad_norm,
    seed,
    model_name,
    device,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_data = TensorDataset(
        torch.tensor(train_x, dtype=torch.float32),
        torch.tensor(train_y, dtype=torch.long),
    )
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    model = make_model(
        model_name, channels=train_x.shape[1], num_classes=int(np.max(train_y)) + 1
    ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    privacy_engine = PrivacyEngine(accountant="rdp")
    model, optimizer, loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        epochs=epochs,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        max_grad_norm=max_grad_norm,
    )

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            if xb.shape[0] == 0:
                continue
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    epsilon = float(privacy_engine.get_epsilon(target_delta))
    noise_multiplier = float(optimizer.noise_multiplier)

    model.eval()
    def predict(array):
        preds = []
        with torch.no_grad():
            for start in range(0, array.shape[0], 512):
                xb = torch.tensor(array[start : start + 512], dtype=torch.float32).to(device)
                preds.append(model(xb).argmax(dim=1).cpu().numpy())
        return np.concatenate(preds)

    with torch.no_grad():
        train_pred = predict(train_x)
        test_pred = predict(test_x)

    return {
        "actual_epsilon": epsilon,
        "noise_multiplier": noise_multiplier,
        "train_accuracy": float(accuracy_score(train_y, train_pred)),
        "test_accuracy": float(accuracy_score(test_y, test_pred)),
        "test_macro_f1": float(f1_score(test_y, test_pred, average="macro", zero_division=0)),
    }


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_mapping(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario",
        "stride",
        "window_id",
        "owner_id",
        "start",
        "end",
        "row_index",
        "split",
        "label",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["scenario"]].append(row)
    out = []
    for scenario in sorted(grouped):
        items = grouped[scenario]
        first = items[0]
        out.append(
            {
                "scenario": scenario,
                "runs": len(items),
                "train_windows": first["train_windows"],
                "test_windows": first["test_windows"],
                "target_epsilon": first["target_epsilon"],
                "target_delta": first["target_delta"],
                "model": first["model"],
                "actual_epsilon_mean": float(np.mean([r["actual_epsilon"] for r in items])),
                "noise_multiplier_mean": float(np.mean([r["noise_multiplier"] for r in items])),
                "train_accuracy_mean": float(np.mean([r["train_accuracy"] for r in items])),
                "test_accuracy_mean": float(np.mean([r["test_accuracy"] for r in items])),
                "test_accuracy_std": float(np.std([r["test_accuracy"] for r in items], ddof=1))
                if len(items) > 1
                else 0.0,
                "test_macro_f1_mean": float(np.mean([r["test_macro_f1"] for r in items])),
            }
        )
    return out


def write_privacy_reports(aggregate_rows, output_dir, mapping_hashes, mapping_path):
    for row in aggregate_rows:
        report = {
            "accountant": "Opacus RDP accountant",
            "sampler": "Opacus private DataLoader",
            "accountant_assumption": "Poisson-like sampling through Opacus private DataLoader",
            "schedule_mode": "fixed_mapping",
            "accounting_unit": "window",
            "epsilon": row["actual_epsilon_mean"],
            "delta": row["target_delta"],
            "mapping_file_sha256": file_sha256(mapping_path),
            "selected_mapping_sha256": mapping_hashes.get(row["scenario"], ""),
            "note": f"UCI HAR raw inertial-signal DP-SGD demo; accountant unit is generated window/example; model={row['model']}.",
        }
        path = output_dir / f"privacy_report_{row['scenario']}.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def write_summary(path, aggregate_rows, args):
    lines = [
        "# UCI HAR Raw-Signal Opacus DP-SGD Demo",
        "",
        "Purpose: deep sequence sanity check for the unit-aware audit.",
        "This is an integration benchmark, not a final SOTA training run.",
        "",
        "Settings:",
        "",
        "```text",
        f"target_epsilon = {args.target_epsilon}",
        f"target_delta = {args.target_delta}",
        f"epochs = {args.epochs}",
        f"batch_size = {args.batch_size}",
        f"learning_rate = {args.learning_rate}",
        f"max_grad_norm = {args.max_grad_norm}",
        f"seeds = {args.seeds}",
        f"model = {args.model} over 9 raw inertial channels",
        f"device = {args.device}",
        "```",
        "",
        "Results:",
        "",
        "| scenario | model | windows | epsilon | acc | macro f1 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            f"| {row['scenario']} | {row['model']} | {row['train_windows']} | "
            f"{row['actual_epsilon_mean']:.3f} | {row['test_accuracy_mean']:.3f} | "
            f"{row['test_macro_f1_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "```text",
            "The same unit-audit distinction applies when the model consumes raw sequence tensors,",
            "not only hand-crafted feature vectors. Overlap still requires event-level conversion;",
            "non-overlap supports direct event-level wording under the fixed mapping.",
            "```",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default=r"data\uci_har\extracted\UCI HAR Dataset",
    )
    parser.add_argument("--output-dir", default=r"reports\uci_har_sequence_dpsgd_demo_001")
    parser.add_argument("--target-epsilon", type=float, default=8.0)
    parser.add_argument("--target-delta", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.35)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seeds", default="17")
    parser.add_argument(
        "--model",
        choices=["tiny_cnn", "small_tcn"],
        default="tiny_cnn",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_x, train_y, train_subjects = load_inertial_split(root, "train")
    test_x, test_y, _ = load_inertial_split(root, "test")
    train_x, test_x = standardize(train_x, test_x)

    _, train_labels, _ = load_split(root, "train")
    train_records = base_records(
        train_subjects, train_labels, "train", "uci_seq_overlap_50pct"
    )
    variants = {
        "uci_seq_overlap_50pct": train_records,
        "uci_seq_non_overlap": non_overlap(train_records, "uci_seq_non_overlap"),
    }

    all_mapping = []
    for records in variants.values():
        all_mapping.extend(records)
    mapping_path = output_dir / "train_mapping_all_variants.csv"
    write_mapping(all_mapping, mapping_path)
    mapping_hashes = {
        scenario: canonical_records_sha256(records)
        for scenario, records in variants.items()
    }

    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda but torch.cuda.is_available() is false")
    rows = []
    for scenario, records in variants.items():
        x_sub, y_sub = subset_x_y(train_x, train_y, records)
        for seed in seeds:
            result = train_private_cnn(
                x_sub,
                y_sub,
                test_x,
                test_y,
                args.target_epsilon,
                args.target_delta,
                args.epochs,
                args.batch_size,
                args.learning_rate,
                args.max_grad_norm,
                seed,
                args.model,
                device,
            )
            rows.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "train_windows": len(records),
                    "test_windows": int(test_x.shape[0]),
                    "target_epsilon": args.target_epsilon,
                    "target_delta": args.target_delta,
                    "model": args.model,
                    "device": device,
                    **result,
                }
            )

    write_csv(rows, output_dir / "runs.csv")
    aggregate_rows = aggregate(rows)
    write_csv(aggregate_rows, output_dir / "aggregate.csv")
    write_privacy_reports(aggregate_rows, output_dir, mapping_hashes, mapping_path)
    write_summary(output_dir / "summary.md", aggregate_rows, args)
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
