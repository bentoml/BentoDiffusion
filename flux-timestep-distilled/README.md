<div align="center">
    <h1 align="center">Serving FLUX.1 models with BentoML</h1>
</div>

This is a BentoML example project, demonstrating how to build an image generation inference API server using the [FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell), a 12 billion parameter rectified flow transformer capable of generating images from text descriptions.

## Prerequisites

To run the Service locally, we recommend you use an Nvidia GPU with at least 32G VRAM.

## Install dependencies

```bash
git clone https://github.com/bentoml/BentoDiffusion.git
cd BentoDiffusion/flux-timestep-distilled

# Recommend Python 3.11
pip install -r requirements.txt
```

## Run the BentoML Service

We have defined a BentoML Service in `service.py`. Run `bentoml serve` in your project directory to start the Service.

```bash
$ bentoml serve

2024-01-18T18:31:49+0800 [INFO] [cli] Starting production HTTP BentoServer from "service:FluxTimestepDistilled" listening on http://localhost:3000 (Press CTRL+C to quit)
Loading pipeline components...: 100%
```

The server is now active at [http://localhost:3000](http://localhost:3000/). You can interact with it using the Swagger UI or in other different ways.

CURL

```bash
curl -X 'POST' \
  'http://localhost:3000/txt2img' \
  -H 'accept: image/*' \
  -H 'Content-Type: application/json' \
  -d '{
  "prompt": "A cinematic shot of a baby racoon wearing an intricate italian priest robe.",
}'
```

BentoML client

```python
import bentoml

with bentoml.SyncHTTPClient("http://localhost:3000") as client:
        result = client.txt2img(
            prompt="A cinematic shot of a baby racoon wearing an intricate italian priest robe.",
        )
```

For custom deployment in your own infrastructure, use [BentoML to generate an OCI-compliant image](https://docs.bentoml.com/en/latest/get-started/packaging-for-deployment.html).
