<div align="center">
    <h1 align="center">Serving Stable Diffusion 3 Medium with BentoML</h1>
</div>

[Stable Diffusion 3 Medium](https://huggingface.co/stabilityai/stable-diffusion-3-medium) is a Multimodal Diffusion Transformer (MMDiT) text-to-image model that features greatly improved performance in image quality, typography, complex prompt understanding, and resource-efficiency.

This is a BentoML example project, demonstrating how to build an image generation inference API server, using the Stable Diffusion 3 Medium model. See [here](https://docs.bentoml.com/en/latest/examples/overview.html) for a full list of BentoML example projects.

## Prerequisites

- Accept the conditions to gain access to [Stable Diffusion 3 Medium on Hugging Face](https://huggingface.co/stabilityai/stable-diffusion-3-medium).
- To run the Service locally, we recommend you use an Nvidia GPU with at least 20G VRAM.

## Install dependencies

```bash
git clone https://github.com/bentoml/BentoDiffusion.git
cd BentoDiffusion/sd3-medium

# Recommend Python 3.11
pip install -r requirements.txt

export HF_TOKEN=<your-api-key>
```

## Run the BentoML Service

We have defined a BentoML Service in `service.py`. Run `bentoml serve` in your project directory to start the Service.

```python
$ bentoml serve .

2024-01-18T18:31:49+0800 [INFO] [cli] Starting production HTTP BentoServer from "service:SD3Medium" listening on http://localhost:3000 (Press CTRL+C to quit)
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
  "prompt": "A cat holding a sign that says hello world",
  "num_inference_steps": 28,
  "guidance_scale": 7.0
}'
```

Python client

```python
import bentoml

with bentoml.SyncHTTPClient("http://localhost:3000") as client:
        result = client.txt2img(
            prompt="A cat holding a sign that says hello world",
            num_inference_steps=28,
            guidance_scale=7.0
        )
```

## Deploy to BentoCloud

After the Service is ready, you can deploy the application to BentoCloud for better management and scalability. [Sign up](https://www.bentoml.com/) if you haven't got a BentoCloud account.

Make sure you have [logged in to BentoCloud](https://docs.bentoml.com/en/latest/scale-with-bentocloud/manage-api-tokens.html).

```bash
bentoml cloud login
```

Create a BentoCloud secret to store the required environment variable and reference it for deployment.

```bash
bentoml secret create huggingface HF_TOKEN=$HF_TOKEN

bentoml deploy --secret huggingface
```

Once the application is up and running on BentoCloud, you can access it via the exposed URL.

**Note**: For custom deployment in your own infrastructure, use [BentoML to generate an OCI-compliant image](https://docs.bentoml.com/en/latest/get-started/packaging-for-deployment.html).
