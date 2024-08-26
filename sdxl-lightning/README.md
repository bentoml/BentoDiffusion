<div align="center">
    <h1 align="center">Serving SDXL Lightning with BentoML</h1>
</div>

This is a BentoML example project, demonstrating how to build an image generation inference API server using the [SDXL-Lightning model](https://huggingface.co/ByteDance/SDXL-Lightning), a lightning-fast text-to-image generation model that is able to generate high-quality 1024px images in a few steps.

See [here](https://github.com/bentoml/BentoML/tree/main/examples) for a full list of BentoML example projects.

## Prerequisites

To run the Service locally, we recommend you use a Nvidia GPU with at least 16G VRAM.

## Install dependencies

```bash
git clone https://github.com/bentoml/BentoDiffusion.git
cd BentoDiffusion/sdxl-lightning

# Recommend Python 3.11
pip install -r requirements.txt
```

## Run the BentoML Service

We have defined a BentoML Service in `service.py`. Run `bentoml serve` in your project directory to start the Service.

```python
$ bentoml serve .

2024-01-18T18:31:49+0800 [INFO] [cli] Starting production HTTP BentoServer from "service:SDXLLightning" listening on http://localhost:3000 (Press CTRL+C to quit)
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
  "num_inference_steps": 1,
  "guidance_scale": 0
}'
```

BentoML client

```python
import bentoml

with bentoml.SyncHTTPClient("http://localhost:3000") as client:
        result = client.txt2img(
            prompt="A cinematic shot of a baby racoon wearing an intricate italian priest robe.",
            num_inference_steps=1,
            guidance_scale=0.0
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
