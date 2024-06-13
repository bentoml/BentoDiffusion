<div align="center">
    <h1 align="center">Serving SDXL + LCM LoRAs with BentoML</h1>
</div>

[Latent Consistency Models (LCM)](https://huggingface.co/papers/2310.04378) introduce a method to improve how images are created, especially with models like Stable Diffusion (SD) and Stable Diffusion XL (SDXL). By integrating LCM LoRAs for SD-based models, you can significantly reduce computational timeframe within just 2 to 8 steps.

This is a BentoML example project, demonstrating how to build a REST API server for SD XL using [LCM LoRAs](https://huggingface.co/blog/lcm_lora). See [here](https://github.com/bentoml/BentoML?tab=readme-ov-file#%EF%B8%8F-what-you-can-build-with-bentoml) for a full list of BentoML example projects.

## Prerequisites

- You have installed Python 3.8+ and `pip`. See the [Python downloads page](https://www.python.org/downloads/) to learn more.
- You have a basic understanding of key concepts in BentoML, such as Services. We recommend you read [Quickstart](https://docs.bentoml.com/en/1.2/get-started/quickstart.html) first.
- (Optional) We recommend you create a virtual environment for dependency isolation for this project. See the [Conda documentation](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or the [Python documentation](https://docs.python.org/3/library/venv.html) for details.

## Install dependencies

```bash
git clone https://github.com/bentoml/BentoLCM.git
cd BentoLCM
pip install -r requirements.txt
```

## Run the BentoML Service

We have defined a BentoML Service in `service.py`. Run `bentoml serve` in your project directory to start the Service.

```bash
bentoml serve .
```

The server is now active at [http://localhost:3000](http://localhost:3000/). You can interact with it using the Swagger UI or in other different ways.

CURL

```bash
curl -X 'POST' \
  'http://localhost:3000/txt2img' \
  -H 'accept: image/*' \
  -H 'Content-Type: application/json' \
  -d '{
  "prompt": "close-up photography of old man standing in the rain at night, in a street lit by lamps, leica 35mm summilux"
}' -o out.jpg
```

Python client

```python
import bentoml
from pathlib import Path

with bentoml.SyncHTTPClient("http://localhost:3000") as client:
    result_path = client.txt2img(
        guidance_scale=1,
        num_inference_steps=4,
        prompt="close-up photography of old man standing in the rain at night, in a street lit by lamps, leica 35mm summilux",
    )

    destination_path = Path("/path/to/save/image.png")
    result_path.rename(destination_path)
```

For detailed explanations of the Service code, see [Stable Diffusion XL with LCM LoRAs](https://docs.bentoml.org/en/latest/use-cases/diffusion-models/sdxl-lcm-lora.html).

## Deploy to BentoCloud

After the Service is ready, you can deploy the application to BentoCloud for better management and scalability. [Sign up](https://www.bentoml.com/) if you haven't got a BentoCloud account.

Make sure you have [logged in to BentoCloud](https://docs.bentoml.com/en/latest/bentocloud/how-tos/manage-access-token.html), then run the following command to deploy it.

```bash
bentoml deploy .
```

Once the application is up and running on BentoCloud, you can access it via the exposed URL.

**Note**: For custom deployment in your own infrastructure, use [BentoML to generate an OCI-compliant image](https://docs.bentoml.com/en/latest/guides/containerization.html).
