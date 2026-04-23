#!/home/hhc/anaconda3/envs/drivestudio/bin/python
"""Build teacher labeling requests and optional fallback final CoC dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nuscenes_coc.exporter import read_json, read_jsonl, write_json, write_jsonl
from nuscenes_coc.teacher_labeling import (
    assemble_final_dataset,
    build_rule_fallback_teacher_responses,
    build_teacher_requests,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build teacher labeling assets from intermediate CoC samples.")
    parser.add_argument("--input", required=True, help="Intermediate CoC JSON path")
    parser.add_argument("--requests-output", required=True, help="Teacher request JSONL output path")
    parser.add_argument("--fallback-output", default=None, help="Optional rule-fallback teacher response JSONL path")
    parser.add_argument("--final-output", default=None, help="Optional final CoC JSON output path")
    parser.add_argument("--teacher-responses", default=None, help="Optional teacher response JSONL path to assemble final dataset")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    samples = read_json(input_path)
    requests = build_teacher_requests(samples)
    write_jsonl(Path(args.requests_output), requests)
    print(f"Wrote {len(requests)} teacher requests to {args.requests_output}")

    fallback_responses = None
    if args.fallback_output:
        fallback_responses = build_rule_fallback_teacher_responses(samples)
        write_jsonl(Path(args.fallback_output), fallback_responses)
        print(f"Wrote {len(fallback_responses)} fallback teacher responses to {args.fallback_output}")

    if args.final_output:
        teacher_responses = None
        if args.teacher_responses:
            teacher_responses = read_jsonl(Path(args.teacher_responses))
        elif fallback_responses is not None:
            teacher_responses = fallback_responses
        final_samples = assemble_final_dataset(samples, teacher_responses)
        write_json(Path(args.final_output), final_samples)
        print(f"Wrote {len(final_samples)} final CoC samples to {args.final_output}")


if __name__ == "__main__":
    main()
