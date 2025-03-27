<div align="center">
    <h1 align="center">Self-host Diffusion Models with BentoML</h1>
</div>

This repository contains a series of BentoML example projects, demonstrating how to deploy different models in [the Stable Diffusion (SD) family](https://huggingface.co/models?other=stable-diffusion), which is specialized in generating and manipulating images or video clips based on text prompts.

See [here](https://docs.bentoml.com/en/latest/examples/overview.html) for a full list of BentoML example projects.

The following guide uses SDXL Turbo as an example.

## Prerequisites

If you want to test the Service locally, we recommend you use an Nvidia GPU with at least 12GB VRAM.

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


## Choose another diffusion model

To deploy a different diffusion model, go to the corresponding subdirectories of this repository.

- [FLUX.1](flux-timestep-distilled/)
- [Stable Diffusion 3 Medium](sd3-medium/)
- [Stable Diffusion 3.5 Large Turbo](sd3.5-large-turbo/)
- [Stable Diffusion 3.5 Large](sd3.5-large/)
- [Stable Diffusion XL Lightning](sdxl-lightning/)
- [Stable Diffusion XL Turbo](sdxl-turbo/)
- [ControlNet](controlnet/)