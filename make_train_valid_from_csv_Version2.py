#!/usr/bin/env python3
# Create train.txt and valid.txt from the provided CSV files.
# Usage:
#   py -3 make_train_valid_from_csv.py --train twitter_training.csv --valid twitter_validation.csv --outdir .

import argparse, csv, sys
from pathlib import Path

POSSIBLE_TEXT_COLS = ["text", "tweet", "tweet_text", "content", "sentence", "utterance", "message", "text_raw"]

def find_text_column(header):
    header_l = [h.lower() for h in header]
    for cand in POSSIBLE_TEXT_COLS:
        if cand in header_l:
            return header[header_l.index(cand)]
    for h in header:
        if 'text' in h.lower() or 'tweet' in h.lower():
            return h
    return None

def extract_column(csv_path, out_path):
    csv_path = Path(csv_path)
    out_path = Path(out_path)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        return 0
    count = 0
    with csv_path.open(newline='', encoding='utf-8') as fh:
        sample = fh.read(4096)
        fh.seek(0)
        dialect = csv.Sniffer().sniff(sample) if sample else csv.get_dialect('excel')
        reader = csv.DictReader(fh, dialect=dialect)
        header = reader.fieldnames or []
        text_col = find_text_column(header)
        if text_col is None:
            print("Could not find a text column. CSV header columns are:", header, file=sys.stderr)
            return -1
        with out_path.open("w", encoding="utf-8") as out:
            for row in reader:
                text = (row.get(text_col) or "").strip()
                if text:
                    out.write(text.replace("\r","").replace("\n"," ").strip() + "\n")
                    count += 1
    return count

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    train_out = outdir / "train.txt"
    valid_out = outdir / "valid.txt"
    print(f"Reading training CSV {args.train} -> {train_out}")
    tcount = extract_column(args.train, train_out)
    if tcount == -1:
        sys.exit(2)
    print(f"Wrote {tcount} lines to {train_out}")
    print(f"Reading validation CSV {args.valid} -> {valid_out}")
    vcount = extract_column(args.valid, valid_out)
    if vcount == -1:
        sys.exit(2)
    print(f"Wrote {vcount} lines to {valid_out}")
    if tcount == 0: print("Warning: train.txt has 0 lines.", file=sys.stderr)
    if vcount == 0: print("Warning: valid.txt has 0 lines.", file=sys.stderr)

if __name__ == "__main__":
    main()