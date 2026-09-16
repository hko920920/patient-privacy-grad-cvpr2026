import argparse
import csv
from pathlib import Path


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def split_owners(value):
    text = str(value).replace("|", ";")
    owners = [item.strip() for item in text.split(";") if item.strip()]
    if not owners:
        raise ValueError("empty owner attribution")
    return owners


def adapt_rows(rows, args):
    out = []
    for idx, row in enumerate(rows):
        if args.owner_ids_col:
            owners = split_owners(row[args.owner_ids_col])
        else:
            owners = [str(row[args.owner_col])]
        window_id = row.get(args.window_id_col, "") if args.window_id_col else ""
        if not window_id:
            window_id = f"{args.scenario}_{idx}"
        out.append(
            {
                "scenario": row.get(args.scenario_col, args.scenario)
                if args.scenario_col
                else args.scenario,
                "stride": row.get(args.stride_col, "") if args.stride_col else "",
                "window_id": window_id,
                "owner_id": owners[0],
                "owner_ids": ";".join(owners),
                "start": int(float(row[args.start_col])),
                "end": int(float(row[args.end_col])),
            }
        )
    return out


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["scenario", "stride", "window_id", "owner_id", "owner_ids", "start", "end"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert a library window/clip index into the unit-audit mapping schema. "
            "This is a schema adapter, not a privacy accountant."
        )
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--owner-col", default="owner_id")
    parser.add_argument("--owner-ids-col", default=None)
    parser.add_argument("--start-col", default="start")
    parser.add_argument("--end-col", default="end")
    parser.add_argument("--window-id-col", default=None)
    parser.add_argument("--scenario-col", default=None)
    parser.add_argument("--stride-col", default=None)
    parser.add_argument("--scenario", default="adapted_mapping")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = read_csv(Path(args.input_csv))
    adapted = adapt_rows(rows, args)
    write_csv(adapted, Path(args.output_csv))
    print(f"Wrote {args.output_csv}")
    print("Schema: scenario, stride, window_id, owner_id, owner_ids, start, end")


if __name__ == "__main__":
    main()
