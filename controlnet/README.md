<div align="center">
    <h1 align="center">Serving SDXL and ControlNet with BentoML</h1>
</div>

ControlNet is a model designed to control image diffusion processes by conditioning them with additional input images, such as canny edges, user sketches, human poses, depth maps, and more. This allows for greater control over image generation by guiding the model with specific inputs, making it easier to generate targeted images.

This is a BentoML example project, demonstrating how to build an image generation inference API server, using the [SDXL model](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) and [the ControlNet model](https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0).

See [here](https://docs.bentoml.com/en/latest/examples/overview.html) for a full list of BentoML example projects.

## Prerequisites

If you want to test the Service locally, we recommend you use a Nvidia GPU with at least 12GB VRAM.

## Install dependencies

```bash
git clone https://github.com/bentoml/BentoDiffusion.git
cd BentoDiffusion/controlnet

# Recommend Python 3.11
pip install -r requirements.txt
```

## Run the BentoML Service

We have defined a BentoML Service in `service.py`. Run `bentoml serve` in your project directory to start the Service.

```python
$ bentoml serve .

2024-01-18T09:43:40+0800 [INFO] [cli] Prometheus metrics for HTTP BentoServer from "service:APIService" can be accessed at http://localhost:3000/metrics.
2024-01-18T09:43:41+0800 [INFO] [cli] Starting production HTTP BentoServer from "service:APIService" listening on http://localhost:3000 (Press CTRL+C to quit)
```

The server is now active at [http://localhost:3000](http://localhost:3000/). You can interact with it using the Swagger UI or in other different ways.

CURL

```bash
curl -X 'POST' \
    'http://localhost:3000/generate' \
    -H 'accept: image/*' \
    -H 'Content-Type: multipart/form-data' \
    -F 'image=@example-image.png;type=image/png' \
	-F 'prompt=A young man walking in a park, wearing jeans.' \
	-F 'negative_prompt=ugly, disfigured, ill-structured, low resolution' \
	-F 'controlnet_conditioning_scale=0.5' \
	-F 'num_inference_steps=25'
```

Python client

```python
import bentoml
from pathlib import Path

with bentoml.SyncHTTPClient("http://localhost:3000") as client:
    result = client.generate(
        image=Path("example-image.png"),
        prompt="A young man walking in a park, wearing jeans.",
        negative_prompt="ugly, disfigured, ill-structure, low resolution",
        controlnet_conditioning_scale=0.5,
        num_inference_steps=25,
    )
```

For detailed explanations of the Service code, see [ControlNet](https://docs.bentoml.org/en/latest/use-cases/diffusion-models/controlnet.html).

## Deploy to BentoCloud

After the Service is ready, you can deploy the application to BentoCloud for better management and scalability. [Sign up](https://www.bentoml.com/) if you haven't got a BentoCloud account.

Make sure you have [logged in to BentoCloud](https://docs.bentoml.com/en/latest/bentocloud/how-tos/manage-access-token.html), then run the following command to deploy it.

```bash
bentoml deploy .
```

Once the application is up and running on BentoCloud, you can access it via the exposed URL.

**Note**: For custom deployment in your own infrastructure, use [BentoML to generate an OCI-compliant image](https://docs.bentoml.com/en/latest/guides/containerization.html).
