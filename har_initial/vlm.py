"""SmolVLM and Qwen3.5-2B helpers shared by the server and the HAR eval."""

from __future__ import annotations

import threading
import time

import torch
from PIL import Image

SMOL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
QWEN_ID = "Qwen/Qwen3.5-2B"
EMBED_ID = "BAAI/bge-small-en-v1.5"
SMOL_PROMPT = "What is the person doing? Answer with only a short action phrase."
FRAME_PROMPT = (
    "The images are in time order, about one second apart. "
    "What is the person doing across these frames? "
    "Answer with only a short action phrase."
)
QWEN_PIXELS = 28 * 28 * 256
ACTION_TEXTS = {
    "calling": "a person is calling on a phone",
    "clapping": "a person is clapping",
    "cycling": "a person is cycling",
    "dancing": "a person is dancing",
    "drinking": "a person is drinking",
    "eating": "a person is eating",
    "fighting": "a person is fighting",
    "hugging": "a person is hugging",
    "laughing": "a person is laughing",
    "listening_to_music": "a person is listening to music",
    "running": "a person is running",
    "sitting": "a person is sitting",
    "sleeping": "a person is sleeping",
    "texting": "a person is texting",
    "using_laptop": "a person is using a laptop",
}
_embed_lock = threading.Lock()
_embed_model = None
_embed_tokenizer = None
_bank_key = None
_bank_vectors = None

torch.set_num_threads(16)


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_smolvlm(device: torch.device):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(SMOL_ID)
    model = AutoModelForImageTextToText.from_pretrained(SMOL_ID, dtype=torch.float32, low_cpu_mem_usage=True)
    model.to(device).eval()
    return model, processor


def load_qwen(device: torch.device):
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    processor = AutoProcessor.from_pretrained(QWEN_ID)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = Qwen3_5ForConditionalGeneration.from_pretrained(QWEN_ID, dtype=dtype, low_cpu_mem_usage=True)
    model.to(device).eval()
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None:
        image_processor.max_pixels = QWEN_PIXELS
        image_processor.min_pixels = 56 * 56
        size = getattr(image_processor, "size", None)
        if isinstance(size, dict):
            size["longest_edge"] = QWEN_PIXELS
            size["shortest_edge"] = 56 * 56
    return model, processor


def canon(text: str) -> str:
    return " ".join(text.lower().replace("_", " ").replace("-", " ").split())


def load_phrase_encoder():
    """Load the phrase embedder on CPU so Qwen can keep the GPU."""
    global _embed_model, _embed_tokenizer
    with _embed_lock:
        if _embed_model is not None:
            return
        from transformers import AutoModel, AutoTokenizer

        _embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_ID)
        _embed_model = AutoModel.from_pretrained(EMBED_ID)
        _embed_model.eval()


def _encode_phrases(texts: list[str]) -> torch.Tensor:
    load_phrase_encoder()
    batch = _embed_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.inference_mode():
        hidden = _embed_model(**batch).last_hidden_state
    mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    return torch.nn.functional.normalize(pooled, dim=1)


def phrase_similarity(left: str, right: str) -> float:
    """Cosine similarity of two action phrases. An empty phrase scores 0."""
    query = canon(left)
    other = canon(right)
    if not query or not other:
        return 0.0
    vectors = _encode_phrases([query, other])
    return round(float(torch.dot(vectors[0], vectors[1])), 4)


def _option_matrix(options: list[dict]) -> torch.Tensor:
    global _bank_key, _bank_vectors
    key = tuple(canon(item["en"]) for item in options)
    if _bank_vectors is not None and _bank_key == key:
        return _bank_vectors
    vectors = _encode_phrases(list(key))
    _bank_vectors = vectors
    _bank_key = key
    return vectors


def rank_text(sentence: str, options: list[dict], limit: int = 3) -> list[dict]:
    """Match a free phrase to activity sentences by text-encoder cosine."""
    query = canon(sentence)
    if not options:
        return []
    if not query:
        return [
            {"id": item["id"], "en": item["en"], "zh": item.get("zh", ""), "score": 0.0}
            for item in options[:limit]
        ]
    bank = _option_matrix(options)
    scores = torch.mv(bank, _encode_phrases([query])[0])
    order = torch.argsort(scores, descending=True).tolist()
    ranked = []
    for index in order[:limit]:
        item = options[index]
        ranked.append(
            {
                "id": item["id"],
                "en": item["en"],
                "zh": item.get("zh", ""),
                "score": round(float(scores[index]), 4),
            }
        )
    return ranked


def strip_think(text: str) -> str:
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.replace("<think>", " ").strip()


def _min_confidence(scores) -> float | None:
    if not scores:
        return None
    values = []
    for score in scores:
        logits = score[0].float()
        values.append(float(torch.softmax(logits, dim=-1).max()))
    return min(values) if values else None


def _prepare_inputs(kind: str, processor, images: list[Image.Image], prompt: str, device: torch.device):
    if kind == "smolvlm":
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, images=images[:1], return_tensors="pt", do_image_splitting=False)
    else:
        content = [{"type": "image", "image": item} for item in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        template_kwargs = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        try:
            text = processor.apply_chat_template(messages, **template_kwargs)
        except Exception:
            template_kwargs.pop("enable_thinking")
            text = processor.apply_chat_template(messages, **template_kwargs)
        inputs = processor(text=[text], images=images, return_tensors="pt", max_pixels=QWEN_PIXELS)
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}


def _frame_list(image: Image.Image | list[Image.Image]) -> list[Image.Image]:
    images = image if isinstance(image, (list, tuple)) else [image]
    prepared = []
    for item in images:
        frame = item.convert("RGB")
        frame.thumbnail((512, 512))
        prepared.append(frame)
    return prepared


def generate(kind: str, model, processor, image: Image.Image | list[Image.Image], prompt: str, max_new_tokens: int) -> dict:
    device = next(model.parameters()).device
    images = _frame_list(image)
    started = time.perf_counter()
    inputs = _prepare_inputs(kind, processor, images, prompt, device)
    prompt_len = int(inputs["input_ids"].shape[-1])
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
    new_ids = output.sequences[0, prompt_len:]
    raw = strip_think(processor.batch_decode([new_ids], skip_special_tokens=True)[0])
    elapsed = (time.perf_counter() - started) * 1000
    return {
        "raw": raw.strip(),
        "confidence": _min_confidence(output.scores),
        "compute_ms": round(elapsed, 1),
    }
