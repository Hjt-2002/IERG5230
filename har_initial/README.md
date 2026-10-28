# 实时人体动作识别

手机浏览器跑 SmolVLM，笔记本跑 Qwen3.5-2B。实时模式每秒存一帧，只保留最近三帧。画面变化很小时沿用当前短语，不跑 SmolVLM。置信度够高时，短语要连续两帧离开当前动作才上传；置信度低则立刻把这几帧交给 Qwen。

仓库里没有视频数据集，也没有虚拟环境。下面按本机实测过的环境说明怎么装、怎么跑。

## 环境要求

- Python 3.13
- NVIDIA 显卡，驱动支持 CUDA 13.0。本机实测是 RTX 5070 Laptop，PyTorch `2.14.0+cu130`
- 内存建议 16 GB 以上。Qwen3.5-2B 以 bfloat16 加载，显存大约占用 5 GB
- [OpenSSL](https://slproweb.com/products/Win32OpenSSL.html) 在 `PATH` 里。服务每次启动会签发一张自签名 HTTPS 证书，手机浏览器才能打开摄像头
- 笔记本和手机在同一个局域网
- 磁盘预留约 10 GB 给两个模型的首次下载。跑 UCF101 / HMDB51 评测时才会再下载视频，UCF101 压缩包约 6.5 GB，HMDB51 测试集约 0.5 GB

手机端不装 Python。SmolVLM-256M 的 4-bit ONNX 由浏览器从 Hugging Face 下载，优先用 WebGPU。

## 安装

在 `har_initial` 目录里：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` 已指定 PyTorch 的 CUDA 13.0 源。装完后确认显卡可用：

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

应看到 `2.14.0+cu130 True`。

## 启动

```powershell
cd har_initial
.\.venv\Scripts\python.exe app.py
```

首次启动会下载 `Qwen/Qwen3.5-2B` 和 `BAAI/bge-small-en-v1.5`。终端打印本机地址和局域网地址，例如 `https://127.0.0.1:8000` 和 `https://<局域网 IP>:8000`。

笔记本打开本机地址。手机打开局域网地址，接受证书警告，再允许摄像头。页面第一次打开还会下载 SmolVLM。

可用环境变量：

| 变量 | 默认 | 作用 |
| --- | --- | --- |
| `HAR_PORT` | `8000` | 服务端口 |
| `HAR_THRESHOLD` | `0.70` | Qwen 与手机短语的余弦相似度门槛 |

页面上的置信度输入默认 `0.5`，低于它会立刻上传。

每次重启都会换一张证书，手机需要重新点一次继续访问。

## 评测

评测脚本会自己下载数据，数据放在 `har_initial/data/`，这个目录不进仓库。

静图 60 张：

```powershell
.\.venv\Scripts\python.exe eval_har.py
```

UCF101 官方 testlist01 和 HMDB51 测试列表，每段中间附近抽 3 帧。全量需要数小时，并占满这块 GPU，跑的时候不要同时开 `app.py`。

```powershell
.\.venv\Scripts\python.exe eval_video.py
.\.venv\Scripts\python.exe eval_clients.py
```

`eval_video.py --limit 4` 可以只跑每个数据集的前 4 段，用来确认环境。结果写在 `eval_results/` 和 `eval_video_results/`。
