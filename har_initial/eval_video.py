"""Zero-shot clip phrases on UCF101 split-1 test and the HMDB51 test list.

Each clip contributes three frames about one second apart. The same frames
feed three branches:

  smol     SmolVLM only. A quiet frame skips the model and keeps the phrase.
  qwen     Qwen3.5-2B sees all three frames once.
  hybrid   The live rule. Low token confidence uploads now. A confident phrase
           that leaves the current action must do so on two frames before the
           frames seen so far go to Qwen.

Accuracy is the clip label nearest to the final phrase in BGE cosine space.
That is not the official video-classification protocol. Latency is compute
inside the branch. Network round-trip is not included.

SmolVLM and Qwen outputs are cached, so the confidence and similarity grids
do not run the models again.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import re
import shutil
import tarfile
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

import policy
import vlm

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "eval_video_results"
FRAME_ROOT = DATA / "frames"
CONF_GRID = (0.3, 0.4, 0.5, 0.6, 0.7)
SIM_GRID = (0.60, 0.65, 0.70, 0.75, 0.80)
SMOL_TOKENS = 16
QWEN_TOKENS = 32


def class_sentence(name: str) -> str:
    spaced = name.replace("_", " ").replace("-", " ")
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", spaced)
    spaced = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", spaced)
    return "a person is " + " ".join(spaced.lower().split())


def clip_id(dataset: str, key: str) -> str:
    return hashlib.sha1(f"{dataset}:{key}".encode("utf-8")).hexdigest()[:16]


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def abs_path(text: str) -> Path:
    path = Path(text)
    if path.is_absolute():
        return path
    return ROOT / path


def choose_indexes(count: int, fps: float) -> list[int]:
    if count <= 0:
        raise RuntimeError("video has no frames")
    duration = count / fps if fps > 0 else 0.0
    if duration >= 2.2 and count >= 3:
        center = count / 2
        gap = fps
        raw = [center - gap, center, center + gap]
    elif count >= 3:
        raw = [count * 0.2, count * 0.5, count * 0.8]
    else:
        raw = [0, 0, 0]
    indexes = [min(count - 1, max(0, int(round(value)))) for value in raw]
    if len(set(indexes)) < min(3, count) and count >= 3:
        indexes = [0, count // 2, count - 1]
    return indexes


def write_jpeg(bgr, dest: Path) -> None:
    import cv2

    if bgr is None or getattr(bgr, "size", 0) == 0:
        raise RuntimeError(f"empty frame for {dest.name}")
    if bgr.ndim == 2:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    image.thumbnail((512, 512))
    dest.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest, "JPEG", quality=72)


def sample_video(video: Path, dests: list[Path]) -> None:
    import cv2

    if all(dest.exists() and dest.stat().st_size > 0 for dest in dests):
        return
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 1:
        fps = 25.0
    try:
        if count <= 0:
            decoded = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                decoded.append(frame)
            if not decoded:
                raise RuntimeError(f"no frames in {video}")
            indexes = choose_indexes(len(decoded), fps)
            images = [decoded[index] for index in indexes]
        else:
            indexes = choose_indexes(count, fps)
            saved: dict[int, object] = {}
            current = 0
            last = max(indexes)
            while current <= last:
                needed = current in indexes and current not in saved
                if needed:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    saved[current] = frame
                elif not cap.grab():
                    break
                current += 1
            if any(index not in saved for index in indexes):
                raise RuntimeError(f"decoded {sorted(saved)} , wanted {indexes} from {video.name}")
            images = [saved[index] for index in indexes]
    finally:
        cap.release()
    for image, dest in zip(images, dests):
        write_jpeg(image, dest)


def read_jsonl(path: Path) -> dict:
    rows: dict = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("error"):
            continue
        if "frame" in row:
            rows[(row["clip"], int(row["frame"]))] = row
        else:
            rows[(row["clip"], int(row["frames"]))] = row
    return rows


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def download(repo: str, filename: str, directory: Path) -> Path:
    from huggingface_hub import hf_hub_download

    directory.mkdir(parents=True, exist_ok=True)
    saved = hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset", local_dir=str(directory))
    return Path(saved)


def frames_ready(clip: dict) -> bool:
    for frame in clip["frames"]:
        path = abs_path(frame)
        if not path.exists() or path.stat().st_size == 0:
            return False
    return True


def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def load_ready_manifest(path: Path, expected: int, skipped: int) -> list[dict] | None:
    clips = load_json(path, None)
    if not isinstance(clips, list) or not clips:
        return None
    if len(clips) + skipped != expected or any(not frames_ready(clip) for clip in clips):
        return None
    return clips


def prepare_ucf() -> list[dict]:
    out = OUT / "ucf101"
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "clips.json"
    split = DATA / "ucf101" / "splits" / "ucfTrainTestlist" / "testlist01.txt"
    if not split.exists():
        archive = download(
            "MichiganNLP/ucf-101",
            "UCF101TrainTestSplits-RecognitionTask.zip",
            DATA / "ucf101",
        )
        shutil.unpack_archive(str(archive), DATA / "ucf101" / "splits")
    listed = []
    for line in split.read_text(encoding="utf-8").splitlines():
        text = line.strip().replace("\\", "/")
        if not text or text.startswith("#"):
            continue
        label = text.split("/", 1)[0]
        listed.append((text, label))
    skipped_path = out / "skipped.json"
    skipped = load_json(skipped_path, [])
    ready = load_ready_manifest(manifest, len(listed), len(skipped))
    if ready is not None:
        return ready
    failed = {item["source"] for item in skipped}
    zip_path = DATA / "ucf101" / "UCF101.zip"
    if not zip_path.exists():
        zip_path = download("MichiganNLP/ucf-101", "UCF101.zip", DATA / "ucf101")
    video_dir = DATA / "ucf101" / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    clips = []
    print(f"ucf101: indexing {zip_path}", flush=True)
    with zipfile.ZipFile(zip_path) as archive:
        members = {}
        for info in archive.infolist():
            parts = info.filename.replace("\\", "/").split("/")
            if len(parts) >= 2 and parts[-1].lower().endswith(".avi"):
                members["/".join(parts[-2:])] = info
        missing = [item for item, _label in listed if item not in members]
        if missing:
            raise RuntimeError(f"UCF101 zip is missing {len(missing)} test clips, e.g. {missing[0]}")
        for number, (item, label) in enumerate(listed, start=1):
            if item in failed:
                continue
            identity = clip_id("ucf101", item)
            video = video_dir / f"{identity}.avi"
            frames = [FRAME_ROOT / "ucf101" / f"{identity}_{index}.jpg" for index in range(3)]
            try:
                if not video.exists() or video.stat().st_size == 0:
                    with archive.open(members[item]) as source, video.open("wb") as target:
                        shutil.copyfileobj(source, target)
                sample_video(video, frames)
            except Exception as exc:  # noqa: BLE001
                print(f"ucf101 skip {item}: {exc}", flush=True)
                skipped.append({"source": item, "error": str(exc)})
                skipped_path.write_text(json.dumps(skipped), encoding="utf-8")
                continue
            clips.append(
                {
                    "id": identity,
                    "label": label,
                    "source": item,
                    "video": rel(video),
                    "frames": [rel(frame) for frame in frames],
                }
            )
            if number % 100 == 0 or number == len(listed):
                print(f"ucf101 frames {number}/{len(listed)}", flush=True)
                manifest.write_text(json.dumps(clips), encoding="utf-8")
    manifest.write_text(json.dumps(clips), encoding="utf-8")
    return clips


def prepare_hmdb() -> list[dict]:
    out = OUT / "hmdb51"
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "clips.json"
    meta_path = DATA / "hmdb51" / "metadata_test.csv"
    if not meta_path.exists():
        meta_path = download("Sina272/hmdb51-v4", "metadata_test.csv", DATA / "hmdb51")
    labels = {}
    with meta_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            labels[Path(row["file_name"]).name] = row["label"]
    skipped_path = out / "skipped.json"
    skipped = load_json(skipped_path, [])
    ready = load_ready_manifest(manifest, len(labels), len(skipped))
    if ready is not None:
        return ready
    failed = {item["source"] for item in skipped}
    tar_path = DATA / "hmdb51" / "shards" / "test-000000.tar"
    if not tar_path.exists():
        tar_path = download("Sina272/hmdb51-v4", "shards/test-000000.tar", DATA / "hmdb51")
    video_dir = DATA / "hmdb51" / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    clips = []
    print(f"hmdb51: reading {tar_path}", flush=True)
    with tarfile.open(tar_path) as archive:
        grouped: dict[str, dict] = {}
        for member in archive.getmembers():
            if not member.isfile():
                continue
            suffix = Path(member.name).suffix.lower()
            grouped.setdefault(Path(member.name).stem, {})[suffix] = member
        pairs = []
        for stem, group in grouped.items():
            meta = {}
            if ".json" in group:
                payload = archive.extractfile(group[".json"])
                if payload is not None:
                    meta = json.loads(payload.read().decode("utf-8"))
            avi = group.get(".avi") or group.get(".mp4")
            filename = Path(str(meta.get("filename") or "")).name
            if not filename and avi is not None:
                filename = Path(avi.name).name
            label = labels.get(filename) or meta.get("label") or ""
            if avi is None or not label:
                continue
            pairs.append((filename, label, avi))
        if len(pairs) != len(labels):
            sample_members = [member.name for member in archive.getmembers()[:6]]
            sample_names = [filename for filename, _label, _member in pairs[:6]]
            raise RuntimeError(
                f"HMDB metadata has {len(labels)} clips but the tar yielded {len(pairs)}. "
                f"members={sample_members} names={sample_names}"
            )
        for number, (filename, label, member) in enumerate(pairs, start=1):
            if filename in failed:
                continue
            identity = clip_id("hmdb51", filename)
            video = video_dir / f"{identity}.avi"
            frames = [FRAME_ROOT / "hmdb51" / f"{identity}_{index}.jpg" for index in range(3)]
            try:
                if not video.exists() or video.stat().st_size == 0:
                    source = archive.extractfile(member)
                    if source is None:
                        raise RuntimeError(f"missing tar member {member.name}")
                    with video.open("wb") as target:
                        shutil.copyfileobj(source, target)
                sample_video(video, frames)
            except Exception as exc:  # noqa: BLE001
                print(f"hmdb51 skip {filename}: {exc}", flush=True)
                skipped.append({"source": filename, "error": str(exc)})
                skipped_path.write_text(json.dumps(skipped), encoding="utf-8")
                continue
            clips.append(
                {
                    "id": identity,
                    "label": label,
                    "source": filename,
                    "video": rel(video),
                    "frames": [rel(frame) for frame in frames],
                }
            )
            if number % 100 == 0 or number == len(pairs):
                print(f"hmdb51 frames {number}/{len(pairs)}", flush=True)
                manifest.write_text(json.dumps(clips), encoding="utf-8")
    manifest.write_text(json.dumps(clips), encoding="utf-8")
    return clips


def warmup(kind: str, model, processor, frames: int) -> None:
    image = Image.new("RGB", (512, 512), (120, 120, 120))
    images = [image] * frames if kind == "qwen" else image
    prompt = vlm.FRAME_PROMPT if kind == "qwen" and frames > 1 else vlm.SMOL_PROMPT
    tokens = QWEN_TOKENS if kind == "qwen" else SMOL_TOKENS
    vlm.generate(kind, model, processor, images, prompt, tokens)


def run_smol(dataset: str, clips: list[dict]) -> None:
    path = OUT / dataset / "smol.jsonl"
    done = read_jsonl(path)
    pending = []
    for clip in clips:
        for index in range(3):
            if (clip["id"], index) not in done:
                pending.append((clip, index))
    if not pending:
        print(f"{dataset} smol: cache has {len(done)} frames", flush=True)
        return
    device = vlm.pick_device()
    print(f"{dataset} smol: {len(pending)} frames on {device}", flush=True)
    model, processor = vlm.load_smolvlm(device)
    try:
        warmup("smolvlm", model, processor, 1)
        started = time.perf_counter()
        for number, (clip, index) in enumerate(pending, start=1):
            image = Image.open(abs_path(clip["frames"][index])).convert("RGB")
            try:
                generated = vlm.generate("smolvlm", model, processor, image, vlm.SMOL_PROMPT, SMOL_TOKENS)
                row = {
                    "clip": clip["id"],
                    "frame": index,
                    "label": clip["label"],
                    "raw": generated["raw"],
                    "confidence": generated["confidence"],
                    "compute_ms": generated["compute_ms"],
                }
            except Exception as exc:  # noqa: BLE001
                row = {
                    "clip": clip["id"],
                    "frame": index,
                    "label": clip["label"],
                    "raw": "",
                    "confidence": None,
                    "compute_ms": None,
                    "error": str(exc),
                }
                gc.collect()
            append_jsonl(path, row)
            if number == 1 or number % 25 == 0 or number == len(pending):
                elapsed = time.perf_counter() - started
                rate = number / elapsed if elapsed else 0
                left = (len(pending) - number) / rate if rate else 0
                print(
                    f"{dataset} smol {number}/{len(pending)} {row['compute_ms']} ms, about {left / 60:.1f} min left",
                    flush=True,
                )
    finally:
        del model, processor
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def run_qwen(dataset: str, clips: list[dict]) -> None:
    path = OUT / dataset / "qwen.jsonl"
    done = read_jsonl(path)
    pending = []
    for clip in clips:
        for count in (1, 2, 3):
            if (clip["id"], count) not in done:
                pending.append((clip, count))
    if not pending:
        print(f"{dataset} qwen: cache has {len(done)} calls", flush=True)
        return
    device = vlm.pick_device()
    print(f"{dataset} qwen: {len(pending)} calls on {device}", flush=True)
    model, processor = vlm.load_qwen(device)
    try:
        warmup("qwen", model, processor, 1)
        warmup("qwen", model, processor, 3)
        started = time.perf_counter()
        for number, (clip, count) in enumerate(pending, start=1):
            images = [Image.open(abs_path(clip["frames"][index])).convert("RGB") for index in range(count)]
            prompt = vlm.FRAME_PROMPT if count > 1 else vlm.SMOL_PROMPT
            try:
                generated = vlm.generate("qwen", model, processor, images, prompt, QWEN_TOKENS)
                row = {
                    "clip": clip["id"],
                    "frames": count,
                    "label": clip["label"],
                    "raw": generated["raw"],
                    "compute_ms": generated["compute_ms"],
                }
            except Exception as exc:  # noqa: BLE001
                row = {
                    "clip": clip["id"],
                    "frames": count,
                    "label": clip["label"],
                    "raw": "",
                    "compute_ms": None,
                    "error": str(exc),
                }
                gc.collect()
            append_jsonl(path, row)
            if number == 1 or number % 25 == 0 or number == len(pending):
                elapsed = time.perf_counter() - started
                rate = number / elapsed if elapsed else 0
                left = (len(pending) - number) / rate if rate else 0
                print(
                    f"{dataset} qwen {number}/{len(pending)} frames={count} {row['compute_ms']} ms, about {left / 60:.1f} min left",
                    flush=True,
                )
    finally:
        del model, processor
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def p95(values: list[float]) -> float | None:
    if not values:
        return None
    return round(float(np.percentile(np.asarray(values, dtype=np.float64), 95, method="linear")), 1)


def pack(rows: list[dict], extra: dict | None = None) -> dict:
    total = len(rows)
    correct = sum(1 for row in rows if row["pred"] == row["label"])
    latencies = [float(row["latency_ms"]) for row in rows]
    result = {
        "correct": correct,
        "total": total,
        "accuracy": round(correct / total, 4) if total else None,
        "mean_latency_ms": round(sum(latencies) / total, 1) if total else None,
        "p95_latency_ms": p95(latencies),
    }
    if extra:
        result.update(extra)
    return result


def score_dataset(dataset: str, clips: list[dict]) -> dict:
    smol_rows = read_jsonl(OUT / dataset / "smol.jsonl")
    qwen_rows = read_jsonl(OUT / dataset / "qwen.jsonl")
    options = [{"id": label, "en": class_sentence(label), "zh": ""} for label in sorted({clip["label"] for clip in clips})]
    vlm.load_phrase_encoder()
    pred_cache: dict[str, tuple[str | None, float | None]] = {}
    sim_cache: dict[tuple[str, str], tuple[float, float]] = {}

    def predict(phrase: str) -> tuple[str | None, float | None]:
        key = vlm.canon(phrase)
        if not key:
            return None, None
        if key not in pred_cache:
            match = vlm.rank_text(phrase, options, limit=1)[0]
            pred_cache[key] = (match["id"], match["score"])
        return pred_cache[key]

    def similarity(left: str, right: str) -> tuple[float, float]:
        key = (vlm.canon(left), vlm.canon(right))
        if not key[0] or not key[1]:
            return 0.0, 0.0
        if key not in sim_cache:
            order = key if key[0] <= key[1] else (key[1], key[0])
            if order in sim_cache:
                sim_cache[key] = sim_cache[order]
            else:
                started = time.perf_counter()
                score = vlm.phrase_similarity(left, right)
                sim_cache[order] = (score, (time.perf_counter() - started) * 1000)
                sim_cache[key] = sim_cache[order]
        return sim_cache[key]

    prepared = []
    for clip in clips:
        frames = []
        for index, frame_path in enumerate(clip["frames"]):
            row = smol_rows.get((clip["id"], index))
            if row is None:
                raise RuntimeError(f"{dataset} is missing SmolVLM frame {clip['id']} #{index}")
            started = time.perf_counter()
            image = Image.open(abs_path(frame_path)).convert("RGB")
            pixels = policy.pixel_signature(image)
            frames.append(
                {
                    "pixels": pixels,
                    "pixel_ms": (time.perf_counter() - started) * 1000,
                    "smol_raw": row.get("raw") or "",
                    "smol_confidence": row.get("confidence"),
                    "smol_ms": float(row.get("compute_ms") or 0.0),
                }
            )
        prepared.append((clip, frames))

    def replay(branch: str, conf: float, sim: float) -> list[dict]:
        rows = []
        for clip, frames in prepared:
            result = policy.simulate(
                branch,
                frames,
                conf,
                sim,
                lambda count, clip_id=clip["id"]: {
                    "raw": (qwen_rows.get((clip_id, count)) or {}).get("raw") or "",
                    "ms": float((qwen_rows.get((clip_id, count)) or {}).get("compute_ms") or 0.0),
                },
                similarity,
            )
            pred, score = predict(result["phrase"])
            rows.append(
                {
                    "id": clip["id"],
                    "label": clip["label"],
                    "phrase": result["phrase"],
                    "pred": pred,
                    "match_score": score,
                    "latency_ms": result["latency_ms"],
                    "uploads": result["uploads"],
                    "skips": result["skips"],
                    "smol_calls": result["smol_calls"],
                }
            )
        return rows

    def diagnostics(rows: list[dict]) -> dict:
        total_frames = len(rows) * 3
        skips = sum(row["skips"] for row in rows)
        uploads = [row["uploads"] for row in rows]
        mistakes = []
        for row in rows:
            if row["pred"] == row["label"]:
                continue
            if len(mistakes) >= 12:
                break
            mistakes.append(
                {"label": row["label"], "pred": row["pred"], "phrase": row["phrase"], "latency_ms": row["latency_ms"]}
            )
        return {
            "mean_uploads": round(sum(uploads) / len(rows), 4) if rows else None,
            "skip_rate": round(skips / total_frames, 4) if total_frames else None,
            "examples": mistakes,
        }

    smol = replay("smol", policy.CONFIDENCE_DEFAULT, policy.SIM_THRESHOLD)
    qwen = replay("qwen", policy.CONFIDENCE_DEFAULT, policy.SIM_THRESHOLD)
    sensitivity = []
    default_hybrid = None
    for confidence in CONF_GRID:
        for similarity_threshold in SIM_GRID:
            hybrid = replay("hybrid", confidence, similarity_threshold)
            extra = diagnostics(hybrid)
            packed = pack(hybrid, extra)
            packed["confidence"] = confidence
            packed["similarity"] = similarity_threshold
            sensitivity.append(packed)
            if confidence == policy.CONFIDENCE_DEFAULT and similarity_threshold == policy.SIM_THRESHOLD:
                default_hybrid = packed
    labels = [clip["label"] for clip in clips]
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return {
        "clips": len(clips),
        "classes": len(counts),
        "min_per_class": min(counts.values()) if counts else 0,
        "max_per_class": max(counts.values()) if counts else 0,
        "smol": pack(smol, diagnostics(smol)),
        "qwen": pack(qwen, diagnostics(qwen)),
        "hybrid": default_hybrid,
        "sensitivity": sensitivity,
    }


def write_csv(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "branch",
        "confidence",
        "similarity",
        "accuracy",
        "correct",
        "total",
        "mean_latency_ms",
        "p95_latency_ms",
        "mean_uploads",
        "skip_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for dataset, body in summary["datasets"].items():
            for branch in ("smol", "qwen"):
                row = body[branch]
                writer.writerow(
                    {
                        "dataset": dataset,
                        "branch": branch,
                        "confidence": "",
                        "similarity": "",
                        "accuracy": row["accuracy"],
                        "correct": row["correct"],
                        "total": row["total"],
                        "mean_latency_ms": row["mean_latency_ms"],
                        "p95_latency_ms": row["p95_latency_ms"],
                        "mean_uploads": row["mean_uploads"],
                        "skip_rate": row["skip_rate"],
                    }
                )
            for row in body["sensitivity"]:
                writer.writerow(
                    {
                        "dataset": dataset,
                        "branch": "hybrid",
                        "confidence": row["confidence"],
                        "similarity": row["similarity"],
                        "accuracy": row["accuracy"],
                        "correct": row["correct"],
                        "total": row["total"],
                        "mean_latency_ms": row["mean_latency_ms"],
                        "p95_latency_ms": row["p95_latency_ms"],
                        "mean_uploads": row["mean_uploads"],
                        "skip_rate": row["skip_rate"],
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="ucf101,hmdb51")
    parser.add_argument("--limit", type=int, default=0, help="Use the first N prepared clips. 0 means the full test list.")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    selected = [item.strip() for item in args.datasets.split(",") if item.strip()]
    builders = {"ucf101": prepare_ucf, "hmdb51": prepare_hmdb}
    unknown = [name for name in selected if name not in builders]
    if unknown:
        raise SystemExit(f"Unknown dataset: {', '.join(unknown)}")
    prepared = {}
    for name in selected:
        clips = builders[name]()
        if args.limit:
            clips = clips[: args.limit]
        prepared[name] = clips
        print(f"{name}: {len(clips)} clips ready", flush=True)
        skipped_path = OUT / name / "skipped.json"
        if skipped_path.exists():
            print(f"{name}: skipped {len(load_json(skipped_path, []))} clips", flush=True)
    if args.prepare_only:
        return
    if not args.score_only:
        for name, clips in prepared.items():
            run_smol(name, clips)
            run_qwen(name, clips)
    datasets = {}
    for name, clips in prepared.items():
        print(f"{name}: scoring", flush=True)
        datasets[name] = score_dataset(name, clips)
    summary = {
        "protocol": {
            "frames": 3,
            "spacing": "1 second around the middle when the clip is at least 2.2 seconds, otherwise spread across the clip",
            "ucf101": "Official recognition testlist01.",
            "hmdb51": "Test list packaged with Sina272/hmdb51-v4 (30 clips per class when complete).",
            "matcher": "BAAI/bge-small-en-v1.5 mean-pool cosine, argmax over class sentences.",
            "latency": "Sum of pixel checks, SmolVLM generates, Qwen generates, and phrase-encoder calls used for that clip. No network round-trip.",
            "pixel_mad_max": policy.PIXEL_MAD_MAX,
            "leave_limit": policy.LEAVE_LIMIT,
            "smol_max_new_tokens": SMOL_TOKENS,
            "qwen_max_new_tokens": QWEN_TOKENS,
            "device": str(vlm.pick_device()),
            "smol_runtime": "PyTorch float32 on this machine. This is not the phone WebGPU timing.",
            "qwen_runtime": "PyTorch bfloat16 on this machine.",
            "note": "A sparse three-frame phrase is not the official video accuracy of either dataset.",
        },
        "limit": args.limit,
        "datasets": datasets,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(OUT / "sensitivity.csv", summary)
    brief = {}
    for name, body in datasets.items():
        brief[name] = {
            "smol": body["smol"]["accuracy"],
            "qwen": body["qwen"]["accuracy"],
            "hybrid": None if body["hybrid"] is None else body["hybrid"]["accuracy"],
            "smol_mean_ms": body["smol"]["mean_latency_ms"],
            "qwen_mean_ms": body["qwen"]["mean_latency_ms"],
            "hybrid_mean_ms": None if body["hybrid"] is None else body["hybrid"]["mean_latency_ms"],
            "smol_p95_ms": body["smol"]["p95_latency_ms"],
            "qwen_p95_ms": body["qwen"]["p95_latency_ms"],
            "hybrid_p95_ms": None if body["hybrid"] is None else body["hybrid"]["p95_latency_ms"],
        }
    print(json.dumps(brief, indent=2), flush=True)


if __name__ == "__main__":
    main()
