# OneDiffusion

OneDiffusion is an open-source one-stop shop for facilitating the deployment of any diffusion models in production. It caters specifically to the needs of diffusion models, supporting both pretrained and fine-tuned diffusion models with LoRA adapters.

Key features include:

- **Broad compatibility**: Support both pretrained and LoRA-adapted diffusion models, providing flexibility in choosing and deploying the appropriate model for various image generation tasks. It currently supports Stable Diffusion (v1.4, v1.5 and v2.0) and Stable Diffusion XL (v1.0) models.
- **Optimized performance and scalability**: Apply the best in class optimizations for serving diffusion models on your behalf.
- **Dynamic LoRA adapter loading**: Dynamically load and unload LoRA adapters on every request, providing greater adaptability and ensuring the models remain responsive to changing inputs and conditions.
- **First-class support for BentoML**: Seamless integration with the BentoML ecosystem, allowing you to build Bentos and push them to BentoCloud or Yatai. 

OneDiffusion is designed for AI application developers who require a robust and flexible platform for deploying diffusion models in production. The platform offers tools and features to fine-tune, serve, deploy, and monitor these models effectively, streamlining the end-to-end workflow for diffusion model deployment.

## Get started

### Prerequisites

You have installed Python 3.8 (or later) and `pip`.

### Install OneDiffusion

Install OneDiffusion by using `pip` as follows:

```
pip install onediffusion
```

To verify the installation, run:

```
$ onediffusion -h

Usage: onediffusion [OPTIONS] COMMAND [ARGS]...

       ____  ____    ____
      / ___||  _ \  / ___|  ___ _ ____   _____ _ __
      \___ \| | | | \___ \ / _ \ '__\ \ / / _ \ '__|
       ___) | |_| |  ___) |  __/ |   \ V /  __/ |
      |____/|____/  |____/ \___|_|    \_/ \___|_|

          An open platform for operating diffusion models in production.
          Fine-tune, serve, deploy, and monitor any diffusion models with ease.


Options:
  -v, --version  Show the version and exit.
  -h, --help     Show this message and exit.

Commands:
  download  Setup LLM interactively.
  start     Start any diffusion models as a REST server.
```

## Start a diffusion server

OneDiffusion allows you to quickly spin up any diffusion models. To start a server, run:

```bash
onediffusion start stable-diffusion
```

This starts a server at http://0.0.0.0:3000/. You can interact with it by visiting the web UI or send a request via `curl`.

Prompt:

```
{
  "prompt": "a bento box",
  "negative_prompt": null,
  "height": 768,
  "width": 768,
  "num_inference_steps": 50,
  "guidance_scale": 7.5,
  "eta": 0
}'
```

Output:



By default, OneDiffusion uses `stabilityai/stable-diffusion-2` to start the server. To use a specific model version, add the `--model-id` option as below:

```bash
onediffusion start stable-diffusion --model-id runwayml/stable-diffusion-v1-5
```

OneDiffusion downloads the models to the BentoML local Model Store if they have not been registered before. To view your models, install BentoML first with `pip install bentoml` and then run:

```
$ bentoml models list

Tag                                                                                         Module                              Size        Creation Time
pt-sd-stabilityai--stable-diffusion-2:1e128c8891e52218b74cde8f26dbfc701cb99d79              bentoml.diffusers                   4.81 GiB    2023-08-16 17:52:33
pt-sdxl-stabilityai--stable-diffusion-xl-base-1.0:bf714989e22c57ddc1c453bf74dab4521acb81d8  bentoml.diffusers                   13.24 GiB   2023-08-16 16:09:01
```

### Add LoRA weights

Low-Rank Adaptation (LoRA) is a training method to fine-tune models without the need to retrain all parameters. You can add LoRA weights to your diffusion models for specific data needs.

Add the `--lora-weights` option as below:

```
onediffusion start stable-diffusion --lora-weights "/path/to/lora-weights.safetensors"
```

A comparison of images with and without LoRA:



### Start a Stable Diffusion XL server

OneDiffusion also supports running Stable Diffusion XL 1.0, the most advanced development in the Stable Diffusion text-to-image suite of models launched by Stability AI. To start an XL server, simply run:

```
onediffusion start stable-diffusion-xl
```

It downloads the model automatically if it does not exist locally. Options such as `--model-id` and `--lora-weights` are also supported. For more information, run `onediffusion start stable-diffusion-xl --help`.

## Download a model

If you want to download a diffusion model without starting a server, use the `onediffusion download` command. For example:

```
onediffusion download stable-diffusion --model-id "CompVis/stable-diffusion-v1-4"
```

## Create a BentoML Runner

You can create a BentoML Runner with the `diffusers_runners.create_runner()` function, which downloads the model specified automatically if it does not exist locally.

```python
import bentoml

# Create a runner for a Stable Diffusion model
runner = bentoml.diffusers_runners.create_runner("CompVis/stable-diffusion-v1-4")

# Create a runner for a Stable Diffusion XL model
runner_xl = bentoml.diffusers_runners.stable_diffusion_xl.create_runner("stabilityai/stable-diffusion-xl-base-1.0")
```

You can then wrap the Runner into a BentoML Service. See the [BentoML documentation](https://docs.bentoml.com/en/latest/concepts/service.html) for more details.

## Build a Bento