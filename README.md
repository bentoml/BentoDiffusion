<div align="center">
    <h1 align="center">Self-host Diffusion Models with BentoML</h1>
</div>

This is a BentoML example project, showing you how to serve and deploy a series of diffusion models in [the Stable Diffusion (SD) family](https://huggingface.co/models?other=stable-diffusion), which is specialized in generating and manipulating images based on text prompts.

See [here](https://github.com/bentoml/BentoML/tree/main/examples) for a full list of BentoML example projects.

The following guide uses SDXL Turbo as an example.

## Prerequisites

- You have installed Python 3.9+ and `pip`. See the [Python downloads page](https://www.python.org/downloads/) to learn more.
- You have a basic understanding of key concepts in BentoML, such as Services. We recommend you read [Quickstart](https://docs.bentoml.com/en/latest/get-started/quickstart.html) first.
- If you want to test the Service locally, a Nvidia GPU with at least 12GB VRAM will boost performance significantly.
- (Optional) We recommend you create a virtual environment for dependency isolation for this project. See the [Conda documentation](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or the [Python documentation](https://docs.python.org/3/library/venv.html) for details.

## Install dependencies

```bash
git clone https://github.com/bentoml/BentoDiffusion.git
cd BentoDiffusion/sdxl-turbo
pip install -r requirements.txt
```

## Run the BentoML Service

We have defined a BentoML Service in `service.py`. Run `bentoml serve` in your project directory to start the Service.

```python
$ bentoml serve .

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

Make sure you have [logged in to BentoCloud](https://docs.bentoml.com/en/latest/bentocloud/how-tos/manage-access-token.html), then run the following command to deploy it.

```bash
bentoml deploy .
```

Once the application is up and running on BentoCloud, you can access it via the exposed URL.

**Note**: For custom deployment in your own infrastructure, use [BentoML to generate an OCI-compliant image](https://docs.bentoml.com/en/latest/guides/containerization.html).


## Choose another diffusion model

To deploy a different diffusion model, go to the corresponding subdirectories of this repository.

- [ControlNet](controlnet/)
- [Latent Consistency Model](lcm/)
- [Stable Diffusion 2 with 4x upscaler](sd2upscaler/)
- [SDXL Lightning](sdxl-lightning/)
- [SDXL Turbo](sdxl-turbo/)
- [Stable Video Diffusion](svd/)