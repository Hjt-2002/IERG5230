"""Several phones run SmolVLM. One laptop runs Qwen.

Each client walks its own clips with the live rule. Pixel checks and SmolVLM
stay on that client, so clients compute at the same time. A Qwen request
enters one FIFO queue, which is the laptop inference lock. Phrase similarity
runs on the server CPU after the lock is released and does not block the next
image.

The clip phrase does not depend on the queue. Accuracy stays at the
single-client hybrid number. Latency grows by the time spent waiting for Qwen.
Service times come from the cached UCF101 and HMDB51 runs.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import random
import time
from collections import deque

import numpy as np
from PIL import Image

import policy
import vlm
from eval_video import OUT, abs_path, class_sentence, read_jsonl

CLIENTS = (1, 2, 4, 8, 16)


def trace_hybrid(frames: list[dict], conf_threshold: float, sim_threshold: float, qwen_at, similarity) -> dict:
    """Same decisions as policy.simulate. Steps are local, server CPU, or server GPU."""
    steps: list[tuple] = []
    segment = ""
    leave = 0
    previous_pixels = None
    for index, frame in enumerate(frames):
        steps.append(("local", float(frame["pixel_ms"])))
        pixels = frame["pixels"]
        if segment and leave == 0 and previous_pixels is not None and policy.pixel_mad(previous_pixels, pixels) < policy.PIXEL_MAD_MAX:
            previous_pixels = pixels
            continue
        previous_pixels = pixels
        steps.append(("local", float(frame["smol_ms"])))
        sentence = policy.clean_phrase(frame.get("smol_raw") or "")
        confidence = frame.get("smol_confidence")
        unsure = confidence is None or float(confidence) < conf_threshold
        if not unsure and segment:
            score, elapsed = similarity(sentence, segment)
            steps.append(("cpu", elapsed))
            if score >= sim_threshold:
                leave = 0
                continue
            leave += 1
            if leave < policy.LEAVE_LIMIT:
                continue
        if not unsure and not segment:
            if sentence:
                segment = sentence
            leave = 0
            continue
        got = qwen_at(index + 1)
        qwen_phrase = policy.clean_phrase(got.get("raw") or "")
        tail = 0.0
        agreement_phone = 0.0
        if sentence or qwen_phrase:
            agreement_phone, elapsed = similarity(sentence, qwen_phrase)
            tail += elapsed
        agreement_prev = 0.0
        if segment and qwen_phrase:
            agreement_prev, elapsed = similarity(segment, qwen_phrase)
            tail += elapsed
        steps.append(("gpu", float(got.get("ms") or 0.0), tail))
        decision = policy.outcome_qwen(
            sentence,
            segment,
            qwen_phrase,
            agreement_phone,
            agreement_prev,
            sim_threshold,
            index + 1,
        )
        segment = decision["final"]
        leave = 0
    return {"phrase": segment, "steps": steps}


def trace_qwen(raw: str, service_ms: float) -> dict:
    return {"phrase": policy.clean_phrase(raw), "steps": [("gpu", float(service_ms or 0.0), 0.0)]}


def _step_sum(steps: list[tuple]) -> float:
    total = 0.0
    for step in steps:
        if step[0] == "gpu":
            total += step[1] + step[2]
        else:
            total += step[1]
    return total


def simulate_queue(client_traces: list[list[dict]]) -> dict:
    """Run clients together. GPU jobs share one FIFO server."""
    clip_i = [0 for _ in client_traces]
    step_i = [0 for _ in client_traces]
    clip_start = [0.0 for _ in client_traces]
    clip_waits = [0.0 for _ in client_traces]
    rows: list[dict] = []
    queue: deque[tuple[int, float, float, float]] = deque()
    gpu_client = None
    gpu_busy = 0.0
    makespan = 0.0
    events: list[tuple[float, int, str, int, float]] = []
    sequence = 0

    def push(when: float, kind: str, client: int, tail: float = 0.0) -> None:
        nonlocal sequence
        heapq.heappush(events, (when, sequence, kind, client, tail))
        sequence += 1

    def start_gpu(now: float) -> None:
        nonlocal gpu_client, gpu_busy
        if gpu_client is not None or not queue:
            return
        client, arrival, service, tail = queue.popleft()
        clip_waits[client] += now - arrival
        gpu_client = client
        gpu_busy += service
        push(now + service, "gpu_done", client, tail)

    def run_client(client: int, now: float) -> None:
        traces = client_traces[client]
        while clip_i[client] < len(traces):
            steps = traces[clip_i[client]]["steps"]
            while step_i[client] < len(steps):
                step = steps[step_i[client]]
                if step[0] == "local" or step[0] == "cpu":
                    now += step[1]
                    step_i[client] += 1
                    continue
                step_i[client] += 1
                queue.append((client, now, step[1], step[2]))
                start_gpu(now)
                return
            clip = traces[clip_i[client]]
            finish = now
            rows.append(
                {
                    "client": client,
                    "label": clip["label"],
                    "phrase": clip["phrase"],
                    "latency_ms": finish - clip_start[client],
                    "wait_ms": clip_waits[client],
                }
            )
            clip_i[client] += 1
            step_i[client] = 0
            clip_waits[client] = 0.0
            clip_start[client] = now
        nonlocal_makespan(now)

    def nonlocal_makespan(now: float) -> None:
        nonlocal makespan
        makespan = max(makespan, now)

    for client in range(len(client_traces)):
        if client_traces[client]:
            push(0.0, "run", client)
    while events:
        now, _seq, kind, client, tail = heapq.heappop(events)
        if kind == "run":
            run_client(client, now)
            continue
        gpu_client = None
        start_gpu(now)
        run_client(client, now + tail)
    return {"rows": rows, "gpu_busy_ms": gpu_busy, "makespan_ms": makespan}


def assign(traces: list[dict], clients: int, seed: int) -> list[list[dict]]:
    order = list(range(len(traces)))
    random.Random(seed).shuffle(order)
    buckets: list[list[dict]] = [[] for _ in range(clients)]
    for index, source in enumerate(order):
        buckets[index % clients].append(traces[source])
    return buckets


def percentile(values: list[float]) -> float | None:
    if not values:
        return None
    return round(float(np.percentile(np.asarray(values, dtype=np.float64), 95, method="linear")), 1)


def summarize(rows: list[dict], gpu_busy: float, makespan: float) -> dict:
    latencies = [float(row["latency_ms"]) for row in rows]
    waits = [float(row["wait_ms"]) for row in rows]
    by_client: dict[int, list[float]] = {}
    for row in rows:
        by_client.setdefault(row["client"], []).append(float(row["latency_ms"]))
    client_means = [sum(values) / len(values) for values in by_client.values() if values]
    total = len(rows)
    return {
        "clips": total,
        "mean_latency_ms": round(sum(latencies) / total, 1) if total else None,
        "p95_latency_ms": percentile(latencies),
        "mean_wait_ms": round(sum(waits) / total, 1) if total else None,
        "p95_wait_ms": percentile(waits),
        "gpu_busy": round(gpu_busy / makespan, 4) if makespan else None,
        "makespan_ms": round(makespan, 1),
        "client_mean_min_ms": round(min(client_means), 1) if client_means else None,
        "client_mean_max_ms": round(max(client_means), 1) if client_means else None,
    }


def build_traces(dataset: str) -> list[dict]:
    clips = json.loads((OUT / dataset / "clips.json").read_text(encoding="utf-8"))
    smol_rows = read_jsonl(OUT / dataset / "smol.jsonl")
    qwen_rows = read_jsonl(OUT / dataset / "qwen.jsonl")
    vlm.load_phrase_encoder()
    sim_cache: dict[tuple[str, str], tuple[float, float]] = {}

    def similarity(left: str, right: str) -> tuple[float, float]:
        key = (vlm.canon(left), vlm.canon(right))
        if not key[0] or not key[1]:
            return 0.0, 0.0
        order = key if key[0] <= key[1] else (key[1], key[0])
        if order not in sim_cache:
            started = time.perf_counter()
            score = vlm.phrase_similarity(left, right)
            sim_cache[order] = (score, (time.perf_counter() - started) * 1000)
        return sim_cache[order]

    traces = []
    started = time.perf_counter()
    for number, clip in enumerate(clips, start=1):
        frames = []
        for index, frame_path in enumerate(clip["frames"]):
            row = smol_rows.get((clip["id"], index))
            if row is None:
                raise RuntimeError(f"{dataset} is missing SmolVLM frame {clip['id']} #{index}")
            image = Image.open(abs_path(frame_path)).convert("RGB")
            pixel_started = time.perf_counter()
            pixels = policy.pixel_signature(image)
            frames.append(
                {
                    "pixels": pixels,
                    "pixel_ms": (time.perf_counter() - pixel_started) * 1000,
                    "smol_raw": row.get("raw") or "",
                    "smol_confidence": row.get("confidence"),
                    "smol_ms": float(row.get("compute_ms") or 0.0),
                }
            )

        def qwen_at(count: int, clip_id: str = clip["id"]) -> dict:
            row = qwen_rows.get((clip_id, count)) or {}
            return {"raw": row.get("raw") or "", "ms": float(row.get("compute_ms") or 0.0)}

        hybrid = trace_hybrid(frames, policy.CONFIDENCE_DEFAULT, policy.SIM_THRESHOLD, qwen_at, similarity)
        full = qwen_rows.get((clip["id"], 3)) or {}
        qwen = trace_qwen(full.get("raw") or "", float(full.get("compute_ms") or 0.0))
        traces.append(
            {
                "id": clip["id"],
                "label": clip["label"],
                "hybrid": hybrid,
                "qwen": qwen,
            }
        )
        if number == 1 or number % 500 == 0 or number == len(clips):
            elapsed = time.perf_counter() - started
            print(f"{dataset} traces {number}/{len(clips)} in {elapsed:.0f}s", flush=True)
    return traces


def predict_map(traces: list[dict], field: str) -> dict[str, str | None]:
    options = [{"id": label, "en": class_sentence(label), "zh": ""} for label in sorted({item["label"] for item in traces})]
    cache: dict[str, str | None] = {}
    for item in traces:
        phrase = item[field]["phrase"]
        key = vlm.canon(phrase)
        if key not in cache:
            if not key:
                cache[key] = None
            else:
                cache[key] = vlm.rank_text(phrase, options, limit=1)[0]["id"]
    return cache


def run_dataset(dataset: str, traces: list[dict], clients: list[int], seed: int) -> list[dict]:
    hybrid_pred = predict_map(traces, "hybrid")
    qwen_pred = predict_map(traces, "qwen")
    reports = []
    for policy_name, field, predicted in (
        ("hybrid", "hybrid", hybrid_pred),
        ("qwen", "qwen", qwen_pred),
    ):
        jobs = [{**item[field], "label": item["label"]} for item in traces]
        truth = {vlm.canon(item[field]["phrase"]): predicted[vlm.canon(item[field]["phrase"])] for item in traces}
        for count in clients:
            result = simulate_queue(assign(jobs, count, seed))
            rows = result["rows"]
            hits = sum(1 for row in rows if truth.get(vlm.canon(row["phrase"])) == row["label"])
            packed = summarize(rows, result["gpu_busy_ms"], result["makespan_ms"])
            packed.update(
                {
                    "dataset": dataset,
                    "policy": policy_name,
                    "clients": count,
                    "correct": hits,
                    "accuracy": round(hits / len(rows), 4) if rows else None,
                }
            )
            reports.append(packed)
            print(
                f"{dataset} {policy_name} clients={count} acc={packed['accuracy']} "
                f"mean={packed['mean_latency_ms']} p95={packed['p95_latency_ms']} "
                f"wait={packed['mean_wait_ms']}",
                flush=True,
            )
    return reports


def _self_check() -> None:
    solid = lambda value: bytes([value]) * (policy.PIXEL_SIDE * policy.PIXEL_SIDE)

    def frame(value, phrase, confidence, smol_ms=200.0):
        return {
            "pixels": solid(value),
            "pixel_ms": 1.0,
            "smol_raw": phrase,
            "smol_confidence": confidence,
            "smol_ms": smol_ms,
        }

    def similarity(left, right):
        same = policy.clean_phrase(left) == policy.clean_phrase(right) and policy.clean_phrase(left) != ""
        return (1.0 if same else 0.1, 5.0)

    def qwen_at(count):
        return {"raw": "cycling", "ms": 100.0}

    frames = [frame(0, "sitting", 0.9), frame(80, "running", 0.9), frame(160, "dancing", 0.9)]
    traced = trace_hybrid(frames, 0.5, 0.7, qwen_at, similarity)
    simulated = policy.simulate("hybrid", frames, 0.5, 0.7, qwen_at, similarity)
    assert traced["phrase"] == simulated["phrase"], (traced["phrase"], simulated["phrase"])
    assert abs(_step_sum(traced["steps"]) - simulated["latency_ms"]) < 1.0, (traced["steps"], simulated["latency_ms"])

    alone = simulate_queue([[{"label": "a", "phrase": "one", "steps": [("local", 50.0), ("gpu", 100.0, 0.0)]}]])
    assert alone["rows"][0]["wait_ms"] == 0.0
    assert abs(alone["rows"][0]["latency_ms"] - 150.0) < 1e-6

    pair = [
        [{"label": "a", "phrase": "one", "steps": [("local", 50.0), ("gpu", 100.0, 0.0)]}],
        [{"label": "b", "phrase": "two", "steps": [("local", 50.0), ("gpu", 100.0, 0.0)]}],
    ]
    shared = simulate_queue(pair)
    by_client = {row["client"]: row for row in shared["rows"]}
    assert abs(by_client[0]["wait_ms"]) < 1e-6 and abs(by_client[0]["latency_ms"] - 150.0) < 1e-6
    assert abs(by_client[1]["wait_ms"] - 100.0) < 1e-6 and abs(by_client[1]["latency_ms"] - 250.0) < 1e-6
    print("client queue self-check ok")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="ucf101,hmdb51")
    parser.add_argument("--clients", default="1,2,4,8,16")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    _self_check()
    selected = [item.strip() for item in args.datasets.split(",") if item.strip()]
    counts = [int(item) for item in args.clients.split(",") if item.strip()]
    reports = []
    for dataset in selected:
        print(f"{dataset}: building traces", flush=True)
        traces = build_traces(dataset)
        reports.extend(run_dataset(dataset, traces, counts, args.seed))
    payload = {
        "seed": args.seed,
        "confidence": policy.CONFIDENCE_DEFAULT,
        "similarity": policy.SIM_THRESHOLD,
        "server": "One FIFO queue. Qwen holds the GPU. Phrase similarity uses the CPU after the GPU is released.",
        "clients": "Clips are shuffled and dealt round-robin. Every client starts at time 0 and runs its clips back to back.",
        "service_time": "Cached laptop PyTorch times from the three-frame eval. SmolVLM is float32. Qwen is bfloat16. This is not phone WebGPU time and does not include the network.",
        "reports": reports,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "clients.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    csv_path = OUT / "clients.csv"
    fields = [
        "dataset",
        "policy",
        "clients",
        "accuracy",
        "correct",
        "clips",
        "mean_latency_ms",
        "p95_latency_ms",
        "mean_wait_ms",
        "p95_wait_ms",
        "gpu_busy",
        "makespan_ms",
        "client_mean_min_ms",
        "client_mean_max_ms",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(reports)
    print(json.dumps(reports, indent=2), flush=True)


if __name__ == "__main__":
    main()
