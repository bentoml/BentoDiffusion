<div align="center">
    <h1 align="center">Serving SV3D (Stable Video 3D) with BentoML</h1>
</div>

[Stable Video 3D (SV3D)](https://huggingface.co/stabilityai/sv3d) is a generative model based on Stable Video Diffusion, fine-tuned for generating orbital videos of an object from a single image input. This allows for the creation of consistent 3D views and can be used for novel view synthesis and 3D generation workflows.

This is a BentoML example project demonstrating how to build an SV3D inference API server. It takes a single image as input and returns a **ZIP file** containing 21 orbital frames and a GIF animation. See [here](https://docs.bentoml.com/en/latest/examples/overview.html) for a full list of BentoML example projects.

## Prerequisites

- Accept the conditions to gain access to [Stable Video 3D (SV3D) on Hugging Face](https://huggingface.co/stabilityai/sv3d).
- To run the Service locally, you need an Nvidia GPU with at least 20G VRAM.

## Install dependencies

```bash
git clone https://github.com/bentoml/BentoDiffusion.git
cd BentoDiffusion/sv3d

# Recommend Python 3.11
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126

# Login to Hugging Face to access the model
huggingface-cli login
```

## Run the BentoML Service

We have defined a BentoML Service in `service.py`. Run `bentoml serve` in your project directory to start the Service.

```bash
$ bentoml serve

2024-01-18T18:31:49+0800 [INFO] [cli] Starting production HTTP BentoServer from "service:SV3DService" listening on http://localhost:3000 (Press CTRL+C to quit)
Loading pipeline components...: 100%
```

The server is now active at [http://localhost:3000](http://localhost:3000/). You can interact with it using the Swagger UI or in other different ways.

CURL

```bash
curl -X POST \
  "http://localhost:3000/generate" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@./test.jpg" \
  -F "elevation_deg=10.0" \
  --output result.zip
```

Python client

```python
import bentoml
from pathlib import Path

with bentoml.SyncHTTPClient("http://localhost:3000") as client:
    result_path = client.generate(
        image=Path("./test.jpg"),
        elevation_deg=10.0
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
