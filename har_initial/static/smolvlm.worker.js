import {
  AutoProcessor,
  AutoModelForVision2Seq,
  RawImage,
  TextStreamer,
  env,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.1";

const MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct";
const PROMPT = "What is the person doing? Answer with only a short action phrase.";

env.allowLocalModels = false;
env.useBrowserCache = typeof caches !== "undefined";
env.backends.onnx.wasm.numThreads = 1;

function wasmThreads() {
  const cores = navigator.hardwareConcurrency || 1;
  if (isIPhone() || !self.crossOriginIsolated || cores < 2) return 1;
  return Math.min(4, cores);
}

let processor = null;
let model = null;
let deviceName = null;

function post(message) {
  self.postMessage(message);
}

const QUANTIZED = {
  embed_tokens: "fp16",
  vision_encoder: "q4",
  decoder_model_merged: "q4",
};

const WEBGPU_DTYPE = {
  embed_tokens: "fp16",
  vision_encoder: "q4f16",
  decoder_model_merged: "q4f16",
};

function isIPhone() {
  const ua = self.navigator?.userAgent || "";
  return /iPhone|iPad|iPod/.test(ua);
}

async function pickDevice() {
  const wasm = { device: "wasm", dtype: QUANTIZED, label: "4-bit WASM" };
  if (!navigator.gpu) return wasm;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return wasm;
    const f16 = adapter.features?.has?.("shader-f16");
    return { device: "webgpu", dtype: f16 ? WEBGPU_DTYPE : QUANTIZED, label: "4-bit WebGPU" };
  } catch {
    return wasm;
  }
}

async function warmup() {
  const image = new RawImage(new Uint8ClampedArray(64 * 64 * 3), 64, 64, 3);
  const messages = [{ role: "user", content: [{ type: "image" }, { type: "text", text: PROMPT }] }];
  const text = processor.apply_chat_template(messages, { add_generation_prompt: true });
  const inputs = await processor(text, [image], { do_image_splitting: false });
  await model.generate({ ...inputs, max_new_tokens: 1, do_sample: false });
}

async function loadModel(choice, progress_callback) {
  return AutoModelForVision2Seq.from_pretrained(MODEL_ID, {
    dtype: choice.dtype,
    device: choice.device,
    progress_callback,
  });
}

async function loadWasm(progress_callback) {
  const wasmChoice = { device: "wasm", dtype: QUANTIZED };
  env.backends.onnx.wasm.numThreads = wasmThreads();
  try {
    return await loadModel(wasmChoice, progress_callback);
  } catch (error) {
    if (env.backends.onnx.wasm.numThreads === 1) throw error;
    env.backends.onnx.wasm.numThreads = 1;
    return await loadModel(wasmChoice, progress_callback);
  }
}

async function load() {
  const choice = await pickDevice();
  deviceName = choice.label;
  post({ status: "loading", device: deviceName });
  const progress_callback = (info) => post({ status: "progress", data: info, device: deviceName });
  processor = await AutoProcessor.from_pretrained(MODEL_ID, { progress_callback });
  if (choice.device === "wasm") {
    model = await loadWasm(progress_callback);
  } else {
    try {
      model = await loadModel(choice, progress_callback);
    } catch {
      deviceName = "4-bit WASM";
      post({
        status: "loading",
        device: deviceName,
        note: "WebGPU could not start the model. Switching to 4-bit WASM.",
      });
      model = await loadWasm(progress_callback);
    }
  }
  if (deviceName.includes("WebGPU")) {
    post({ status: "loading", device: deviceName, note: "Compiling the vision encoder…" });
    try {
      await warmup();
    } catch {
      // The first photo compiles the shaders if this pass fails.
    }
  }
  post({ status: "ready", device: deviceName });
}

function argmaxProbability(score) {
  let data;
  try {
    data = score?.data ?? score;
  } catch {
    return null;
  }
  if (!data || data.length === 0) return null;
  let maxLogit = -Infinity;
  for (let i = 0; i < data.length; i += 1) {
    if (data[i] > maxLogit) maxLogit = data[i];
  }
  let sumExp = 0;
  for (let i = 0; i < data.length; i += 1) sumExp += Math.exp(data[i] - maxLogit);
  if (!Number.isFinite(sumExp) || sumExp <= 0) return null;
  return 1 / sumExp;
}

function blobToDataUrl(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(blob);
  });
}

const MAX_NEW_TOKENS = 16;
// The vision encoder always resizes to 512 and emits 64 tokens. A smaller JPEG is scaled back up.
const MAX_IMAGE_SIDE = 512;

async function fitImage(image) {
  const longest = Math.max(image.width, image.height);
  if (longest <= MAX_IMAGE_SIDE) return image;
  const scale = MAX_IMAGE_SIDE / longest;
  const width = Math.max(1, Math.round(image.width * scale));
  const height = Math.max(1, Math.round(image.height * scale));
  return image.resize(width, height);
}

function since(mark) {
  return Math.round(performance.now() - mark);
}

async function generate(blob) {
  if (!model || !processor) throw new Error("SmolVLM is not loaded");
  const started = performance.now();
  let mark = started;
  const decoded = typeof RawImage.fromBlob === "function"
    ? await RawImage.fromBlob(blob)
    : await RawImage.fromURL(await blobToDataUrl(blob));
  const decodeMs = since(mark);
  mark = performance.now();
  const image = await fitImage(decoded);
  const fitMs = since(mark);
  mark = performance.now();
  const messages = [
    {
      role: "user",
      content: [
        { type: "image" },
        { type: "text", text: PROMPT },
      ],
    },
  ];
  const text = processor.apply_chat_template(messages, { add_generation_prompt: true });
  const templateMs = since(mark);
  mark = performance.now();
  post({ status: "generate", phase: "image" });
  const inputs = await processor(text, [image], { do_image_splitting: false });
  const prepareMs = since(mark);
  const pieces = [];
  let raw = "";
  let tokenCount = 0;
  const tokenMs = [];
  let tokenMark = 0;
  const streamer = new TextStreamer(processor.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (chunk) => {
      raw += chunk;
      if (chunk && chunk.trim()) pieces.push(chunk);
    },
    token_callback_function: (tokens) => {
      const now = performance.now();
      tokenMs.push(Math.round(now - tokenMark));
      tokenMark = now;
      const batch = Array.isArray(tokens) ? tokens.length : 1;
      tokenCount += batch;
      post({
        status: "generate",
        phase: "tokens",
        count: tokenCount,
        max: MAX_NEW_TOKENS,
        text: raw.trim(),
      });
    },
  });
  const tokenProbs = [];
  const recordLogits = (_inputIds, logits) => {
    const probability = argmaxProbability(logits);
    if (probability != null) tokenProbs.push(probability);
    return logits;
  };
  mark = performance.now();
  tokenMark = mark;
  try {
    await model.generate({
      ...inputs,
      max_new_tokens: MAX_NEW_TOKENS,
      do_sample: false,
      streamer,
      logits_processor: [recordLogits],
    });
  } catch {
    raw = "";
    pieces.length = 0;
    tokenCount = 0;
    tokenMs.length = 0;
    tokenProbs.length = 0;
    tokenMark = performance.now();
    const output = await model.generate({
      ...inputs,
      max_new_tokens: MAX_NEW_TOKENS,
      do_sample: false,
      streamer,
    });
    if (!raw && output) {
      const decoded = processor.batch_decode(output.sequences ?? output, { skip_special_tokens: true });
      raw = Array.isArray(decoded) ? decoded[0] : String(decoded);
    }
  }
  const generateMs = since(mark);
  mark = performance.now();
  const confidence = tokenProbs.length ? Math.min(...tokenProbs) : null;
  const confidenceMs = since(mark);
  post({
    status: "result",
    raw: raw.trim(),
    tokens: pieces,
    confidence,
    confidenceKnown: confidence != null,
    device: deviceName,
    compute_ms: since(started),
    timing: {
      decode_ms: decodeMs,
      fit_ms: fitMs,
      template_ms: templateMs,
      prepare_ms: prepareMs,
      generate_ms: generateMs,
      token_ms: tokenMs,
      confidence_ms: confidenceMs,
      total_ms: since(started),
    },
  });
}

self.addEventListener("message", (event) => {
  const { type, blob } = event.data || {};
  if (type === "load") {
    load().catch((error) => post({ status: "error", data: String(error) }));
    return;
  }
  if (type === "generate") {
    generate(blob).catch((error) => post({ status: "error", data: String(error) }));
  }
});
