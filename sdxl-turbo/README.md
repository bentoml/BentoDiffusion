<div align="center">
    <h1 align="center">Serving SDXL Turbo with BentoML</h1>
</div>

[Stable Diffusion XL Turbo](https://huggingface.co/stabilityai/sdxl-turbo) is a real-time text-to-image generation model utilizing a novel distillation technique called Adversarial Diffusion Distillation (ADD). This technology enables SDXL Turbo to generate images in a single step, significantly enhancing performance and reducing computational requirements without sacrificing image quality.

This is a BentoML example project, demonstrating how to build an image generation inference API server, using the SDXL Turbo model. See [here](https://docs.bentoml.com/en/latest/examples/overview.html) for a full list of BentoML example projects.

## Prerequisites

To run the Service locally, we recommend you use an Nvidia GPU with at least 12G VRAM.

## Install dependencies

```bash
git clone https://github.com/bentoml/BentoDiffusion.git
cd BentoDiffusion/sdxl-turbo

# Recommend Python 3.11
pip install -r requirements.txt
```

## Run the BentoML Service

We have defined a BentoML Service in `service.py`. Run `bentoml serve` in your project directory to start the Service.

```bash
$ bentoml serve

2024-01-18T18:31:49+0800 [INFO] [cli] Starting production HTTP BentoServer from "service:SDXLTurboService" listening on http://localhost:3000 (Press CTRL+C to quit)
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

Python client

```python
import bentoml

with bentoml.SyncHTTPClient("http://localhost:3000") as client:
        result = client.txt2img(
            prompt="A cinematic shot of a baby racoon wearing an intricate italian priest robe.",
            num_inference_steps=1,
            guidance_scale=0.0
        )
```

For detailed explanations of the Service code, see [Stable Diffusion XL Turbo](https://docs.bentoml.com/en/latest/use-cases/diffusion-models/sdxl-turbo.html).

## Deploy to BentoCloud

After the Service is ready, you can deploy the application to BentoCloud for better management and scalability. [Sign up](https://www.bentoml.com/) if you haven't got a BentoCloud account.

Make sure you have [logged in to BentoCloud](https://docs.bentoml.com/en/latest/scale-with-bentocloud/manage-api-tokens.html).

```bash
bentoml cloud login
```

Deploy it to BentoCloud.

```bash
bentoml deploy
```

Once the application is up and running on BentoCloud, you can access it via the exposed URL.

**Note**: For custom deployment in your own infrastructure, use [BentoML to generate an OCI-compliant image](https://docs.bentoml.com/en/latest/get-started/packaging-for-deployment.html).
