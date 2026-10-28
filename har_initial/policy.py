"""Accept, skip, and segment rules shared by the server and the video eval.

The phone page keeps the same constants. Health returns them so the page can
pick up a change without editing two files.
"""

from __future__ import annotations

from PIL import Image

PIXEL_SIDE = 32
PIXEL_MAD_MAX = 10.0
LEAVE_LIMIT = 2
CONFIDENCE_DEFAULT = 0.5
SIM_THRESHOLD = 0.70


def decide(agreement: float, threshold: float, qwen_phrase: str) -> str:
    """Keep the phone phrase when Qwen's phrase is empty or close in meaning."""
    if not qwen_phrase:
        return "weak"
    if agreement >= threshold:
        return "confirmed"
    return "corrected"


def outcome_text(sentence: str, previous: str, agreement: float | None, threshold: float) -> dict:
    same = bool(previous) and agreement is not None and agreement >= threshold
    return {
        "uploaded_image": False,
        "frame_count": 0,
        "status": "extended" if same else "text_only",
        "accepted": True,
        "same_as_segment": same,
        "agreement": agreement,
        "phone_phrase": sentence,
        "qwen_phrase": None,
        "final": previous if same else sentence,
        "threshold": threshold,
    }


def outcome_qwen(
    sentence: str,
    previous: str,
    qwen_phrase: str,
    agreement_phone: float,
    agreement_prev: float,
    threshold: float,
    frame_count: int,
) -> dict:
    if previous and qwen_phrase and agreement_prev >= threshold:
        status = "held"
        agreement = agreement_prev
        final = previous
        same = True
    else:
        agreement = agreement_phone
        status = decide(agreement, threshold, qwen_phrase)
        final = qwen_phrase if status == "corrected" else sentence
        same = False
    return {
        "uploaded_image": True,
        "frame_count": frame_count,
        "status": status,
        "accepted": status == "confirmed",
        "same_as_segment": same,
        "agreement": agreement,
        "phone_phrase": sentence,
        "qwen_phrase": qwen_phrase,
        "final": final,
        "threshold": threshold,
    }


def clean_phrase(text: str) -> str:
    """Match the phrase cleanup the phone page applies before posting."""
    cleaned = " ".join(str(text or "").split()).strip()
    lowered = cleaned.lower()
    for prefix in ("assistant:", "caption:"):
        if lowered.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
            lowered = cleaned.lower()
    for marker in ("<end_of_utterance>", "<|endoftext|>", "<|im_end|>"):
        cleaned = cleaned.replace(marker, " ")
    cleaned = " ".join(cleaned.split()).strip()
    if len(cleaned) >= 2 and cleaned[0] == '"' and cleaned[-1] == '"':
        cleaned = cleaned[1:-1].strip()
    return cleaned


def pixel_signature(image: Image.Image) -> bytes:
    gray = image.convert("L").resize((PIXEL_SIDE, PIXEL_SIDE), Image.Resampling.BILINEAR)
    return gray.tobytes()


def pixel_mad(left: bytes, right: bytes) -> float:
    if not left or not right or len(left) != len(right):
        return 1e9
    total = sum(abs(a - b) for a, b in zip(left, right))
    return total / len(left)


def simulate(
    branch: str,
    frames: list[dict],
    conf_threshold: float,
    sim_threshold: float,
    qwen_at,
    similarity,
) -> dict:
    """Replay one clip.

    frames carry pixels, pixel_ms, smol_raw, smol_confidence, and smol_ms.
    qwen_at(count) returns the cached Qwen result for the first count frames.
    similarity(left, right) returns (score, milliseconds).
    Latency is the compute the branch would spend. Class matching is separate.
    """
    if branch == "qwen":
        got = qwen_at(len(frames))
        return {
            "phrase": clean_phrase(got.get("raw") or ""),
            "latency_ms": round(float(got.get("ms") or 0.0), 1),
            "uploads": 1,
            "skips": 0,
            "smol_calls": 0,
        }

    segment = ""
    leave = 0
    previous_pixels = None
    total = 0.0
    uploads = 0
    skips = 0
    smol_calls = 0
    for index, frame in enumerate(frames):
        total += float(frame["pixel_ms"])
        pixels = frame["pixels"]
        if segment and leave == 0 and previous_pixels is not None and pixel_mad(previous_pixels, pixels) < PIXEL_MAD_MAX:
            previous_pixels = pixels
            skips += 1
            continue
        previous_pixels = pixels
        total += float(frame["smol_ms"])
        smol_calls += 1
        sentence = clean_phrase(frame.get("smol_raw") or "")
        confidence = frame.get("smol_confidence")
        if branch == "smol":
            if sentence:
                segment = sentence
            continue
        unsure = confidence is None or float(confidence) < conf_threshold
        if not unsure and segment:
            score, elapsed = similarity(sentence, segment)
            total += elapsed
            if score >= sim_threshold:
                leave = 0
                continue
            leave += 1
            if leave < LEAVE_LIMIT:
                continue
        if not unsure and not segment:
            if sentence:
                segment = sentence
            leave = 0
            continue
        got = qwen_at(index + 1)
        total += float(got.get("ms") or 0.0)
        uploads += 1
        qwen_phrase = clean_phrase(got.get("raw") or "")
        agreement_phone = 0.0
        if sentence or qwen_phrase:
            agreement_phone, elapsed = similarity(sentence, qwen_phrase)
            total += elapsed
        agreement_prev = 0.0
        if segment and qwen_phrase:
            agreement_prev, elapsed = similarity(segment, qwen_phrase)
            total += elapsed
        decision = outcome_qwen(
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
    return {
        "phrase": segment,
        "latency_ms": round(total, 1),
        "uploads": uploads,
        "skips": skips,
        "smol_calls": smol_calls,
    }


def _self_check() -> None:
    solid = lambda value: bytes([value]) * (PIXEL_SIDE * PIXEL_SIDE)

    def frame(value, phrase, confidence, smol_ms=200.0):
        return {
            "pixels": solid(value),
            "pixel_ms": 1.0,
            "smol_raw": phrase,
            "smol_confidence": confidence,
            "smol_ms": smol_ms,
        }

    def similarity(left, right):
        same = clean_phrase(left) == clean_phrase(right) and clean_phrase(left) != ""
        return (1.0 if same else 0.1, 5.0)

    quiet = [frame(0, "sitting", 0.9), frame(0, "running", 0.9), frame(0, "cycling", 0.9)]
    smol = simulate("smol", quiet, 0.5, 0.7, lambda count: {"raw": "cycling", "ms": 100}, similarity)
    assert smol["smol_calls"] == 1 and smol["skips"] == 2 and smol["phrase"] == "sitting", smol

    calls = []

    def qwen_at(count):
        calls.append(count)
        return {"raw": "cycling", "ms": 100}

    changed = [frame(0, "sitting", 0.9), frame(80, "running", 0.9), frame(160, "dancing", 0.9)]
    hybrid = simulate("hybrid", changed, 0.5, 0.7, qwen_at, similarity)
    assert calls == [3] and hybrid["uploads"] == 1 and hybrid["phrase"] == "cycling", (calls, hybrid)

    calls.clear()
    returned = [frame(0, "sitting", 0.9), frame(80, "running", 0.9), frame(160, "sitting", 0.9)]
    held = simulate("hybrid", returned, 0.5, 0.7, qwen_at, similarity)
    assert calls == [] and held["phrase"] == "sitting", (calls, held)

    calls.clear()
    unsure = [frame(0, "sitting", 0.9), frame(80, "running", 0.2)]
    early = simulate("hybrid", unsure, 0.5, 0.7, qwen_at, similarity)
    assert calls == [2] and early["phrase"] == "cycling", (calls, early)

    qwen = simulate("qwen", changed, 0.5, 0.7, lambda count: {"raw": "cycling", "ms": 80}, similarity)
    assert qwen["latency_ms"] == 80.0 and qwen["phrase"] == "cycling" and qwen["smol_calls"] == 0, qwen
    print("policy self-check ok")


if __name__ == "__main__":
    _self_check()
