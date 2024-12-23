<div align="center">
    <h1 align="center">Serving Stable Video Diffusion with BentoML</h1>
</div>

[Stable Video Diffusion (SVD)](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid) is a foundation model for generative video based on the image model Stable Diffusion. It comes in the form of two primary image-to-video models, SVD and SVD-XT, capable of generating 14 and 25 frames at customizable frame rates between 3 and 30 frames per second.

This is a BentoML example project, demonstrating how to build a video generation inference API server, using the SVD model. See [here](https://docs.bentoml.com/en/latest/examples/overview.html) for a full list of BentoML example projects.

## Prerequisites

To run this project locally, we recommend you use a Nvidia GPU with 16G+ VRAM.

## Install dependencies

```bash
git clone https://github.com/bentoml/BentoDiffusion.git
cd BentoDiffusion/svd

# Recommend Python 3.11
pip install -r requirements.txt
```

## Run the BentoML Service

We have defined a BentoML Service in `service.py`. Run `bentoml serve` in your project directory to start the Service. Skip to cloud deployment if you don't have a Nvidia GPU locally.

```python
$ bentoml serve .

2024-01-19T07:29:04+0800 [INFO] [cli] Starting production HTTP BentoServer from "service:SVDService" listening on http://localhost:3000 (Press CTRL+C to quit)
Loading pipeline components...: 100%
```

The server is now active. Open your browser at [http://localhost:3000](http://localhost:3000/) to interact via the web UI, or use an HTTP API client to call the local endpoint:

CURL

```bash
curl -X 'POST' \
  'http://localhost:3000/generate' \
  -H 'accept: */*' \
  -H 'Content-Type: multipart/form-data' \
  -F 'image=@assets/sample.png;type=image/png' \
  -F 'decode_chunk_size=2' \
  -F 'seed=null' \
  -o generated.mp4
```

Python client

```python
import bentoml
from pathlib import Path

with bentoml.SyncHTTPClient("http://localhost:3000/") as client:
    result = client.generate(
        decode_chunk_size=2,
        image=@assets/sample.png,
        seed=0,
    )
```

## Deploy to BentoCloud

After the Service is ready, you can deploy the application to BentoCloud for better management and scalability. [Sign up](https://www.bentoml.com/) if you haven't got a BentoCloud account.

Make sure you have [logged in to BentoCloud](https://docs.bentoml.com/en/latest/bentocloud/how-tos/manage-access-token.html), then run the following command to deploy it.

```bash
bentoml deploy .
```

Once the application is up and running on BentoCloud, you can access it via the exposed URL.

**Note**: For custom deployment in your own infrastructure, use [BentoML to generate an OCI-compliant image](https://docs.bentoml.com/en/latest/guides/containerization.html).
