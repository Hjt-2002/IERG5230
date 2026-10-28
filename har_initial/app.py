"""Phone runs SmolVLM. The laptop server runs Qwen3.5-2B.

The browser stores one frame per second and keeps the last three. A frame
close to the previous one keeps the ongoing phrase and skips SmolVLM.
A confident new phrase must leave that action on two frames before the
recent frames, oldest first, go to Qwen. Low token confidence uploads
on the current frame.
"""

from __future__ import annotations

import io
import os
import threading
import traceback
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

import policy
import vlm

ROOT = Path(__file__).resolve().parent
HOST = os.environ.get("HAR_HOST", "0.0.0.0")
PORT = int(os.environ.get("HAR_PORT", "8000"))
THRESHOLD = float(os.environ.get("HAR_THRESHOLD", "0.70"))

app = FastAPI(title="HAR semantic draft")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.middleware("http")
async def cross_origin_isolation(request, call_next):
    """Let desktop browsers run the WASM fallback on more than one thread."""
    response = await call_next(request)
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "credentialless"
    return response

state: dict = {
    "ready": False,
    "error": None,
    "device": None,
    "draft_model": None,
}
infer_lock = threading.Lock()
qwen_model = None
qwen_processor = None


def read_image(payload: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(payload)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}") from exc
    image.thumbnail((768, 768))
    return image


def require_ready():
    if state["error"]:
        raise HTTPException(status_code=500, detail=state["error"])
    if not state["ready"]:
        raise HTTPException(status_code=503, detail="Models are still loading")


def startup_load():
    global qwen_model, qwen_processor
    try:
        device = vlm.pick_device()
        state["device"] = str(device)
        print(f"使用设备: {state['device']}", flush=True)
        qwen_model, qwen_processor = vlm.load_qwen(device)
        state["draft_model"] = vlm.QWEN_ID
        print(f"Qwen 已加载: {vlm.QWEN_ID}", flush=True)
        vlm.load_phrase_encoder()
        print(f"语义相似度已加载: {vlm.EMBED_ID}", flush=True)
        state["ready"] = True
    except Exception:
        state["error"] = traceback.format_exc()
        print(state["error"], flush=True)


@app.on_event("startup")
def on_startup():
    threading.Thread(target=startup_load, daemon=True).start()


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/health")
def health():
    return {
        "ready": state["ready"],
        "error": state["error"],
        "device": state["device"],
        "draft_model": state["draft_model"],
        "embed_model": vlm.EMBED_ID,
        "threshold": THRESHOLD,
        "pixel_mad_max": policy.PIXEL_MAD_MAX,
        "leave_limit": policy.LEAVE_LIMIT,
    }


def _uploads(frames: list[UploadFile] | UploadFile | None) -> list[UploadFile]:
    if frames is None:
        return []
    if isinstance(frames, UploadFile):
        return [frames]
    return [item for item in frames if item is not None]


@app.post("/api/clip")
async def qwen_check(
    sentence: str = Form(...),
    previous: str = Form(""),
    frames: list[UploadFile] | None = File(None),
):
    """Keep the ongoing action, or replace it after Qwen sees the recent frames."""
    require_ready()
    sentence = " ".join(sentence.split()).strip()
    previous = " ".join(previous.split()).strip()
    if not sentence:
        raise HTTPException(status_code=400, detail="Missing sentence")
    images = []
    for item in _uploads(frames)[:3]:
        if not item.filename:
            continue
        images.append(read_image(await item.read()))

    def run():
        import time

        begin = time.perf_counter()
        if not images:
            agreement = vlm.phrase_similarity(sentence, previous) if previous else None
            elapsed = (time.perf_counter() - begin) * 1000
            result = policy.outcome_text(sentence, previous, agreement, THRESHOLD)
            result["compute_ms"] = round(elapsed, 1)
            return result
        prompt = vlm.FRAME_PROMPT if len(images) > 1 else vlm.SMOL_PROMPT
        with infer_lock:
            generated = vlm.generate(
                "qwen",
                qwen_model,
                qwen_processor,
                images,
                prompt,
                max_new_tokens=32,
            )
        qwen_phrase = " ".join(generated["raw"].split()).strip()
        agreement_phone = vlm.phrase_similarity(sentence, qwen_phrase)
        agreement_prev = vlm.phrase_similarity(previous, qwen_phrase) if previous else 0.0
        elapsed = (time.perf_counter() - begin) * 1000
        result = policy.outcome_qwen(
            sentence,
            previous,
            qwen_phrase,
            agreement_phone,
            agreement_prev,
            THRESHOLD,
            len(images),
        )
        result["compute_ms"] = round(elapsed, 1)
        return result

    return await _to_thread(run)


async def _to_thread(fn):
    import asyncio

    try:
        return await asyncio.to_thread(fn)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def lan_ips() -> list[str]:
    """IPv4 addresses other devices on this network can use. The default route comes first."""
    import socket

    ordered: list[str] = []

    def add(ip: str) -> None:
        if ip and ip not in ordered and not ip.startswith("127.") and not ip.startswith("169.254."):
            ordered.append(ip)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        add(sock.getsockname()[0])
    except OSError:
        pass
    finally:
        sock.close()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            add(info[4][0])
    except OSError:
        pass
    return ordered


def refresh_certificate(ips: list[str]) -> tuple[Path, Path]:
    """Rewrite the HTTPS certificate so its name matches the current addresses."""
    import subprocess

    cert_dir = ROOT / "certs"
    cert_dir.mkdir(exist_ok=True)
    names = ["127.0.0.1", *ips]
    alt_lines = [f"IP.{index} = {ip}" for index, ip in enumerate(names, start=1)]
    alt_lines.append("DNS.1 = localhost")
    common_name = ips[0] if ips else "localhost"
    config = "\n".join(
        [
            "[req]",
            "distinguished_name = req_distinguished_name",
            "x509_extensions = v3_req",
            "prompt = no",
            "",
            "[req_distinguished_name]",
            f"CN = {common_name}",
            "",
            "[v3_req]",
            "subjectAltName = @alt_names",
            "basicConstraints = CA:FALSE",
            "keyUsage = digitalSignature, keyEncipherment",
            "",
            "[alt_names]",
            *alt_lines,
            "",
        ]
    )
    config_path = cert_dir / "openssl.cnf"
    config_path.write_text(config, encoding="ascii")
    cert = cert_dir / "cert.pem"
    key = cert_dir / "key.pem"
    tmp_cert = cert_dir / "cert.pem.new"
    tmp_key = cert_dir / "key.pem.new"
    result = subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-sha256",
            "-days",
            "365",
            "-nodes",
            "-keyout",
            str(tmp_key),
            "-out",
            str(tmp_cert),
            "-config",
            str(config_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        tmp_cert.unlink(missing_ok=True)
        tmp_key.unlink(missing_ok=True)
        detail = (result.stderr or result.stdout or "openssl failed").strip()
        raise RuntimeError(detail)
    tmp_cert.replace(cert)
    tmp_key.replace(key)
    return cert, key


def print_access_urls(ips: list[str]) -> None:
    scheme = "https"
    print(f"本机访问: {scheme}://127.0.0.1:{PORT}", flush=True)
    if not ips:
        print("没有发现局域网地址，其他设备现在连不上。", flush=True)
        return
    print("其他设备访问:", flush=True)
    for ip in ips:
        print(f"  {scheme}://{ip}:{PORT}", flush=True)


if __name__ == "__main__":
    import uvicorn

    ips = lan_ips()
    try:
        cert, key = refresh_certificate(ips)
        print("证书已按当前地址重新签发。手机上需要重新点一次继续访问。", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"证书签发失败，继续使用已有证书: {exc}", flush=True)
        cert = ROOT / "certs" / "cert.pem"
        key = ROOT / "certs" / "key.pem"
    print_access_urls(ips)
    ssl_kwargs = {}
    if cert.exists() and key.exists():
        ssl_kwargs = {"ssl_certfile": str(cert), "ssl_keyfile": str(key)}
    uvicorn.run(app, host=HOST, port=PORT, reload=False, **ssl_kwargs)
