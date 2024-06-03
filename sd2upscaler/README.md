<div align="center">
    <h1 align="center">Serving Stable Diffusion 2 + Upscaler with BentoML</h1>
</div>

This is a BentoML example project, demonstrating how to build an image generation inference API server using the [SD2 model](https://huggingface.co/stabilityai/stable-diffusion-2) and the [upscaler model](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler). See [here](https://github.com/bentoml/BentoML?tab=readme-ov-file#%EF%B8%8F-what-you-can-build-with-bentoml) for a full list of BentoML example projects.

## Prerequisites

- You have installed Python 3.8+ and `pip`. See the [Python downloads page](https://www.python.org/downloads/) to learn more.
- You have a basic understanding of key concepts in BentoML, such as Services. We recommend you read [Quickstart](https://docs.bentoml.com/en/latest/get-started/quickstart.html) first.
- If you want to test this Service locally, we highly recommend using a Nvidia GPU with more than 32G VRAM.
- (Optional) We recommend you create a virtual environment for dependency isolation for this project. See the [Conda documentation](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or the [Python documentation](https://docs.python.org/3/library/venv.html) for details.

## Install dependencies

```bash
git clone https://github.com/bentoml/BentoSD2Upscaler.git
cd BentoSD2Upscaler
pip install -r requirements.txt
```

## Run the BentoML Service

We have defined a BentoML Service in `service.py`. Run `bentoml serve` in your project directory to start the Service.

```bash
$ bentoml serve .

2024-01-19T06:16:28+0800 [INFO] [cli] Starting production HTTP BentoServer from "service:SD2Service" listening on http://localhost:3000 (Press CTRL+C to quit)
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
  "prompt": "photo of a majestic sunrise in the mountains, best quality, 4k",
  "negative_prompt": "low quality, bad quality, sketches",
  "height": 512,
  "width": 512,
  "num_inference_steps": 50,
  "guidance_scale": 7.5,
  "upscale": true
}'
```

Python client

```python
import bentoml

with bentoml.SyncHTTPClient("http://localhost:3000") as client:
    result = client.txt2img(
        guidance_scale=7.5,
        height=512,
        negative_prompt="low quality, bad quality, sketches",
        num_inference_steps=50,
        prompt="photo a majestic sunrise in the mountains, best quality, 4k",
        upscale=True,
        width=512,
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
