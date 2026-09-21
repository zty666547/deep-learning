"""Generate representative test-set saliency maps for the CNN baseline."""

from __future__ import annotations

import argparse

from microc_foundation.explainability import generate_test_saliency


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate predicted-class input-gradient saliency maps."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--per-class", type=int, default=1)
    arguments = parser.parse_args()

    manifest = generate_test_saliency(
        arguments.dataset,
        arguments.checkpoint,
        arguments.output_dir,
        per_class=arguments.per_class,
    )
    print(f"Saved {len(manifest)} saliency maps to {arguments.output_dir}")
    for item in manifest:
        print(
            f"{item['structure_id']} | true={item['true_class']} "
            f"| predicted={item['predicted_class']} | correct={item['correct']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
