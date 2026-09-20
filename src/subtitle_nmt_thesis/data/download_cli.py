import argparse
from pathlib import Path
from subtitle_nmt_thesis.data.dataset_loader import (
    load_parallel_samples,
    export_to_jsonl,
    export_to_srt,
)


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Download educational parallel dataset samples")
    parser.add_argument("--dataset", type=str, default="nptel", choices=["nptel", "iitb"])
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="data/benchmarks")
    return parser.parse_args(args)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    print(f"Fetching {args.num_samples} samples from {args.dataset} ({args.split} split)...")
    samples = load_parallel_samples(
        dataset_name=args.dataset, split=args.split, num_samples=args.num_samples
    )
    print(f"Retrieved {len(samples)} parallel pairs.")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Export jsonl
    jsonl_path = out_dir / f"{args.dataset}_{args.split}.jsonl"
    export_to_jsonl(samples, jsonl_path)
    print(f"Exported JSONL to {jsonl_path}")

    # Export reference SRTs
    src_srt = out_dir / f"{args.dataset}_{args.split}_en.srt"
    tgt_srt = out_dir / f"{args.dataset}_{args.split}_hi.srt"
    export_to_srt(samples, src_srt, tgt_srt)
    print(f"Exported benchmark SRTs:\n  - {src_srt}\n  - {tgt_srt}")


if __name__ == "__main__":
    main()
