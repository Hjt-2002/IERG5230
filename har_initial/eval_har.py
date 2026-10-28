"""Zero-shot check on Bingsu/Human_Action_Recognition.

Three policies share the same stratified sample:
  smolvlm  SmolVLM writes a phrase, then cosine matches it to a class sentence.
  qwen     Qwen3.5-2B writes a phrase, then cosine matches it to a class sentence.
  mixed    Keep SmolVLM when its weakest token is at least the threshold.
           Otherwise use Qwen.

The published test split has placeholder labels, so the sample comes from
the labeled train split. The models are not trained on it.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path

from PIL import Image

import vlm

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "eval_results"
DATASET = "Bingsu/Human_Action_Recognition"
LABEL_ALIASES = {"listeningtomusic": "listening_to_music"}


def class_options(names: list[str]) -> list[dict]:
    options = []
    for name in names:
        key = LABEL_ALIASES.get(name, name)
        english = vlm.ACTION_TEXTS.get(key)
        if english is None:
            english = "a person is " + name.replace("_", " ")
        options.append({"id": name, "en": english, "zh": ""})
    return options


def choose_samples(dataset, per_class: int, seed: int) -> list[dict]:
    names = list(dataset.features["labels"].names)
    groups: dict[int, list[int]] = {label_id: [] for label_id in range(len(names))}
    for index, label in enumerate(dataset["labels"]):
        groups[int(label)].append(index)
    rng = random.Random(seed)
    chosen = []
    for label_id, indices in groups.items():
        rng.shuffle(indices)
        for index in indices[:per_class]:
            chosen.append({"index": index, "label": names[label_id]})
    rng.shuffle(chosen)
    return chosen


def load_jsonl(path: Path) -> dict[int, dict]:
    rows = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[int(row["index"])] = row
    return rows


def rescore_file(path: Path, options: list[dict]) -> dict[int, dict]:
    """Rematch cached phrases with text-encoder cosine. Generation is left as stored."""
    rows: dict[int, dict] = {}
    ordered = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        raw = " ".join(str(row.get("raw") or "").split()).strip()
        if raw and not row.get("error"):
            match = vlm.rank_text(raw, options, limit=1)[0]
            row["pred"] = match["id"]
            row["match_score"] = match["score"]
        else:
            row["pred"] = None
            row["match_score"] = None
        ordered.append(row)
        rows[int(row["index"])] = row
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered),
        encoding="utf-8",
    )
    return rows


def append_jsonl(path: Path, row: dict):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def frame_of(dataset, index: int) -> Image.Image:
    image = dataset[index]["image"]
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    return image.convert("RGB")


def run_model(kind: str, dataset, samples: list[dict], options: list[dict], path: Path):
    done = load_jsonl(path)
    pending = [sample for sample in samples if sample["index"] not in done]
    if not pending:
        print(f"{kind}: cache already has {len(done)} rows", flush=True)
        return
    device = vlm.pick_device()
    print(f"{kind}: loading on {device}, {len(pending)} images left", flush=True)
    if kind == "smolvlm":
        model, processor = vlm.load_smolvlm(device)
    else:
        model, processor = vlm.load_qwen(device)
    prompt = vlm.SMOL_PROMPT
    limit = 32
    try:
        for number, sample in enumerate(pending, start=1):
            image = frame_of(dataset, sample["index"])
            try:
                generated = vlm.generate(kind, model, processor, image, prompt, limit)
                pred = vlm.rank_text(generated["raw"], options, limit=1)[0]["id"]
                row = {
                    "index": sample["index"],
                    "label": sample["label"],
                    "raw": generated["raw"],
                    "pred": pred,
                    "confidence": generated["confidence"],
                    "compute_ms": generated["compute_ms"],
                }
            except Exception as exc:  # noqa: BLE001
                row = {
                    "index": sample["index"],
                    "label": sample["label"],
                    "raw": "",
                    "pred": None,
                    "confidence": None,
                    "compute_ms": None,
                    "error": str(exc),
                }
            append_jsonl(path, row)
            mark = "ok" if row["pred"] == row["label"] else "miss"
            print(
                f"{kind} {number}/{len(pending)} {row['label']} -> {row['pred']} {mark} {row['compute_ms']} ms",
                flush=True,
            )
    finally:
        del model, processor
        gc.collect()


def accuracy(rows: list[dict], key: str) -> dict:
    total = len(rows)
    correct = sum(1 for row in rows if row.get(key) == row["label"])
    by_label: dict[str, list[int]] = {}
    for row in rows:
        bucket = by_label.setdefault(row["label"], [0, 0])
        bucket[1] += 1
        if row.get(key) == row["label"]:
            bucket[0] += 1
    return {
        "correct": correct,
        "total": total,
        "accuracy": round(correct / total, 4) if total else None,
        "per_class": {name: {"correct": hit, "total": count} for name, (hit, count) in sorted(by_label.items())},
    }


def mix_rows(samples: list[dict], smol: dict[int, dict], qwen: dict[int, dict], threshold: float) -> list[dict]:
    rows = []
    for sample in samples:
        phone = smol[sample["index"]]
        server = qwen[sample["index"]]
        confidence = phone.get("confidence")
        use_qwen = confidence is None or confidence < threshold
        pred = server.get("pred") if use_qwen and server.get("pred") else phone.get("pred")
        rows.append(
            {
                "index": sample["index"],
                "label": sample["label"],
                "pred": pred,
                "smol_pred": phone.get("pred"),
                "qwen_pred": server.get("pred"),
                "confidence": confidence,
                "used_qwen": bool(use_qwen and server.get("pred")),
            }
        )
    return rows


def summarize(samples, smol, qwen, threshold: float) -> dict:
    ordered_smol = [smol[sample["index"]] | {"label": sample["label"]} for sample in samples]
    ordered_qwen = [qwen[sample["index"]] | {"label": sample["label"]} for sample in samples]
    for row, sample in zip(ordered_smol, samples):
        row["pred"] = smol[sample["index"]].get("pred")
    mixed = mix_rows(samples, smol, qwen, threshold)
    sweeps = {}
    for value in (0.3, 0.5, 0.7):
        rows = mix_rows(samples, smol, qwen, value)
        routed = sum(1 for row in rows if row["used_qwen"])
        sweeps[str(value)] = {
            **accuracy(rows, "pred"),
            "routed_to_qwen": routed,
            "routed_rate": round(routed / len(rows), 4) if rows else None,
        }
    return {
        "dataset": DATASET,
        "split": "train",
        "reason": "The published test split labels are placeholders.",
        "samples": len(samples),
        "seed": None,
        "device": str(vlm.pick_device()),
        "smol_model": vlm.SMOL_ID,
        "qwen_model": vlm.QWEN_ID,
        "qwen_max_pixels": vlm.QWEN_PIXELS,
        "threshold": threshold,
        "matcher": {
            "method": "cosine",
            "embed_model": vlm.EMBED_ID,
            "text": "Each phrase is matched to the nearest class activity sentence.",
        },
        "smolvlm": accuracy(ordered_smol, "pred"),
        "qwen": accuracy(ordered_qwen, "pred"),
        "mixed": sweeps[str(threshold)] if str(threshold) in sweeps else sweeps["0.5"],
        "mixed_thresholds": sweeps,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    from datasets import load_dataset

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"loading {DATASET}", flush=True)
    dataset = load_dataset(DATASET, split="train")
    sample_path = OUT / f"samples_c{args.per_class}_s{args.seed}.json"
    if sample_path.exists():
        samples = json.loads(sample_path.read_text(encoding="utf-8"))
    else:
        samples = choose_samples(dataset, args.per_class, args.seed)
        sample_path.write_text(json.dumps(samples), encoding="utf-8")
    if args.limit:
        samples = samples[: args.limit]
    names = sorted({sample["label"] for sample in samples})
    options = class_options(names)
    print(f"{len(samples)} images, {len(names)} classes", flush=True)

    run_model("smolvlm", dataset, samples, options, OUT / "smolvlm.jsonl")
    run_model("qwen", dataset, samples, options, OUT / "qwen.jsonl")
    smol = rescore_file(OUT / "smolvlm.jsonl", options)
    qwen = rescore_file(OUT / "qwen.jsonl", options)
    summary = summarize(samples, smol, qwen, args.threshold)
    summary["seed"] = args.seed
    summary["per_class"] = args.per_class
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "smolvlm": summary["smolvlm"]["accuracy"],
        "qwen": summary["qwen"]["accuracy"],
        "mixed": summary["mixed"]["accuracy"],
        "mixed_routed_rate": summary["mixed"]["routed_rate"],
        "thresholds": {key: value["accuracy"] for key, value in summary["mixed_thresholds"].items()},
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
