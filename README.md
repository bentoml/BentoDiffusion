# OneDiffusion

OneDiffusion is an open-source one-stop shop for facilitating the deployment of any diffusion models in production. It caters specifically to the needs of diffusion models, supporting both pretrained and fine-tuned diffusion models with LoRA adapters.

Key features include:

- 🌐 **Broad compatibility**: Support both pretrained and LoRA-adapted diffusion models, providing flexibility in choosing and deploying the appropriate model for various image generation tasks. It currently supports Stable Diffusion (v1.4, v1.5 and v2.0) and Stable Diffusion XL (v1.0) models. Support for more models (for example, ControlNet) is on the way.
- 💪 **Optimized performance and scalability**: Apply the best in class optimizations for serving diffusion models on your behalf.
- ⌛️ **Dynamic LoRA adapter loading**: Dynamically load and unload LoRA adapters on every request, providing greater adaptability and ensuring the models remain responsive to changing inputs and conditions.
- 🍱 **First-class support for BentoML**: Seamless integration with the [BentoML](https://github.com/bentoml/BentoML) ecosystem, allowing you to build Bentos and push them to [BentoCloud](https://www.bentoml.com/cloud) or [Yatai](https://github.com/bentoml/Yatai). 

OneDiffusion is designed for AI application developers who require a robust and flexible platform for deploying diffusion models in production. The platform offers tools and features to fine-tune, serve, deploy, and monitor these models effectively, streamlining the end-to-end workflow for diffusion model deployment.

## Get started

### Prerequisites

You have installed Python 3.8 (or later) and `pip`.

### Install OneDiffusion

Install OneDiffusion by using `pip` as follows:

```bash
pip install onediffusion
```

To verify the installation, run:

```bash
$ onediffusion -h

Usage: onediffusion [OPTIONS] COMMAND [ARGS]...

 ██████╗ ███╗   ██╗███████╗██████╗ ██╗███████╗███████╗██╗   ██╗███████╗██╗ ██████╗ ███╗   ██╗
██╔═══██╗████╗  ██║██╔════╝██╔══██╗██║██╔════╝██╔════╝██║   ██║██╔════╝██║██╔═══██╗████╗  ██║
██║   ██║██╔██╗ ██║█████╗  ██║  ██║██║█████╗  █████╗  ██║   ██║███████╗██║██║   ██║██╔██╗ ██║
██║   ██║██║╚██╗██║██╔══╝  ██║  ██║██║██╔══╝  ██╔══╝  ██║   ██║╚════██║██║██║   ██║██║╚██╗██║
╚██████╔╝██║ ╚████║███████╗██████╔╝██║██║     ██║     ╚██████╔╝███████║██║╚██████╔╝██║ ╚████║
 ╚═════╝ ╚═╝  ╚═══╝╚══════╝╚═════╝ ╚═╝╚═╝     ╚═╝      ╚═════╝ ╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝
 
          An open platform for operating diffusion models in production.
          Fine-tune, serve, deploy, and monitor any diffusion models with ease.


Options:
  -v, --version  Show the version and exit.
  -h, --help     Show this message and exit.

Commands:
  download  Setup diffusion model interactively.
  start     Start any diffusion models as a REST server.
```

## Start a diffusion server

OneDiffusion allows you to quickly spin up any diffusion models. To start a server, run:

```bash
onediffusion start stable-diffusion
```

This starts a server at http://0.0.0.0:3000/. You can interact with it by visiting the web UI or send a request via `curl`.

```bash
curl -X 'POST' \
  'http://0.0.0.0:3000/text2img' \
  -H 'accept: image/jpeg' \
  -H 'Content-Type: application/json' \
  --output output.jpg \
  -d '{
  "prompt": "a bento box",
  "negative_prompt": null,
  "height": 768,
  "width": 768,
  "num_inference_steps": 50,
  "guidance_scale": 7.5,
  "eta": 0
}'
```

By default, OneDiffusion uses `stabilityai/stable-diffusion-2` to start the server. To use a specific model version, add the `--model-id` option as below:

```bash
onediffusion start stable-diffusion --model-id runwayml/stable-diffusion-v1-5
```

OneDiffusion downloads the models to the BentoML local Model Store if they have not been registered before. To view your models, install BentoML first with `pip install bentoml` and then run:

```bash
$ bentoml models list

Tag                                                                                         Module                              Size        Creation Time
pt-sd-stabilityai--stable-diffusion-2:1e128c8891e52218b74cde8f26dbfc701cb99d79              bentoml.diffusers                   4.81 GiB    2023-08-16 17:52:33
pt-sdxl-stabilityai--stable-diffusion-xl-base-1.0:bf714989e22c57ddc1c453bf74dab4521acb81d8  bentoml.diffusers                   13.24 GiB   2023-08-16 16:09:01
```

### Start a Stable Diffusion XL server

OneDiffusion also supports running Stable Diffusion XL 1.0, the most advanced development in the Stable Diffusion text-to-image suite of models launched by Stability AI. To start an XL server, simply run:

```bash
onediffusion start stable-diffusion-xl
```

It downloads the model automatically if it does not exist locally. Options such as `--model-id` are also supported. For more information, run `onediffusion start stable-diffusion-xl --help`. 

Similarly, visit http://0.0.0.0:3000/ or send a request via `curl` to interact with the XL server. Example prompt:

```
{
  "prompt": "the scene is a picturesque environment with beautiful flowers and trees. In the center, there is a small cat. The cat is shown with its chin being scratched. It is crouched down peacefully. The cat's eyes are filled with excitement and satisfaction as it uses its small paws to hold onto the food, emitting a content purring sound.",
  "negative_prompt": null,
  "height": 1024,
  "width": 1024,
  "num_inference_steps": 50,
  "guidance_scale": 7.5,
  "eta": 0
}
```

Example output:

[sdxl-cat](/example-images/sdxl-cat.jpeg)

### Add LoRA weights

Low-Rank Adaptation (LoRA) is a training method to fine-tune models without the need to retrain all parameters. You can add LoRA weights to your diffusion models for specific data needs.

Add the `--lora-weights` option as below:

```bash
onediffusion start stable-diffusion-xl --lora-weights "/path/to/lora-weights.safetensors"
```

Alternatively, dynamically load LoRA weights by adding the `lora_weights` field:

```
{
  "prompt": "the scene is a picturesque environment with beautiful flowers and trees. In the center, there is a small cat. The cat is shown with its chin being scratched. It is crouched down peacefully. The cat's eyes are filled with excitement and satisfaction as it uses its small paws to hold onto the food, emitting a content purring sound.",
  "negative_prompt": null,
  "height": 1024,
  "width": 1024,
  "num_inference_steps": 50,
  "guidance_scale": 7.5,
  "eta": 0,
  "lora_weights": "/path/to/lora-weights.safetensors"
}
```

Example output:

[dynamic loading](/example-images/dynamic-loading.gif)

## Download a model

If you want to download a diffusion model without starting a server, use the `onediffusion download` command. For example:

```
onediffusion download stable-diffusion --model-id "CompVis/stable-diffusion-v1-4"
```

## Create a BentoML Runner

You can create a BentoML Runner with `diffusers_simple.stable_diffusion.create_runner()`, which downloads the model specified automatically if it does not exist locally.

```python
import bentoml

# Create a Runner for a Stable Diffusion model
runner = bentoml.diffusers_simple.stable_diffusion.create_runner("CompVis/stable-diffusion-v1-4")

# Create a Runner for a Stable Diffusion XL model
runner_xl = bentoml.diffusers_simple.stable_diffusion_xl.create_runner("stabilityai/stable-diffusion-xl-base-1.0")
```

You can then wrap the Runner into a BentoML Service. See the [BentoML documentation](https://docs.bentoml.com/en/latest/concepts/service.html) for more details.

## Build a Bento

A [Bento](https://docs.bentoml.com/en/latest/concepts/bento.html) in BentoML is a deployable artifact with all the source code, models, data files, and dependency configurations. You can build a Bento for a supported diffusion model directly by running `onediffusion build`.

```bash
# Build a Bento with a Stable Diffusion model 
onediffusion build stable-diffusion

# Build a Bento with a Stable Diffusion XL model 
onediffusion build stable-diffusion-xl
```

To specify the model to be packaged into the Bento, use `--model-id`. Otherwise, OneDiffusion packages the default model into the Bento. If the model does not exist locally, OneDiffusion downloads the model automatically.

Once your Bento is ready, you can push it to [BentoCloud](https://www.bentoml.com/cloud) or [Yatai](https://github.com/bentoml/Yatai).

## Roadmap

We are working to improve OneDiffusion in the following ways and invite anyone who is interested in the project to participate 🤝.

- Support more models, such as ControlNet and DeepFloyd IF
- Support more pipelines, such as inpainting
- Add a Python API client to interact with diffusion models
- Implement advanced optimization like AITemplate
- Offer a unified fine-tuning training API

## Contribution

We weclome contributions of all kinds to the OneDiffusion project! Check out the following resources to start your OneDiffusion journey and stay tuned for more announcements about OneDiffusion and BentoML.

- Submit a pull request or create an issue in the [OneDiffusion GitHub repository](https://github.com/bentoml/OneDiffusion).
- Join the [BentoML community on Slack](https://l.bentoml.com/join-slack).
- Follow us on [Twitter](https://twitter.com/bentomlai) and [Linkedin](https://www.linkedin.com/company/bentoml/).