import os
import math
import zipfile
import tempfile
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import bentoml
import pydantic
from omegaconf import OmegaConf
from torchvision.transforms import ToTensor
from einops import repeat, rearrange
from sgm.util import instantiate_from_config


class SV3DArgs(pydantic.BaseModel):
    gpu_count: int = 1
    timeout: int = 1200

    model_id: str = "stabilityai/sv3d"
    config_file: str = "sv3d_p.yaml"
    weights_file: str = "sv3d_p.safetensors"

    num_frames: int = 21
    num_steps: int = 50
    decode_chunk_size: int = 1

    default_elevation: float = 10.0
    default_fps_id: int = 6
    default_motion_bucket_id: int = 127
    default_cond_aug: float = 1e-5

    @property
    def image(self) -> bentoml.images.Image:
        return (
            bentoml.images.Image(python_version="3.11")
            .requirements_file("requirements.txt")
            .system_packages("ffmpeg", "git")
        )


sv3d_args = bentoml.use_arguments(SV3DArgs)


def _get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def _load_model_architecture(
    config_path: str, model_path: str, device: str, num_frames: int, num_steps: int
):
    config = OmegaConf.load(config_path)

    if device == "cuda":
        if hasattr(config.model.params, "conditioner_config"):
            config.model.params.conditioner_config.params.emb_models[
                0
            ].params.open_clip_embedding_config.params.init_device = device

    config.model.params.ckpt_path = model_path
    config.model.params.sampler_config.params.verbose = False
    config.model.params.sampler_config.params.num_steps = num_steps
    config.model.params.sampler_config.params.guider_config.params.num_frames = int(
        num_frames
    )

    with torch.device(device) if device == "cuda" else torch.no_grad():
        model = instantiate_from_config(config.model).to(device).eval()

    return model


def _get_batch(keys, value_dict, N, T, device):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to(device),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames" or key == "cond_frames_without_noise":
            batch[key] = repeat(value_dict[key], "1 ... -> b ...", b=N[0])
        elif key == "polars_rad" or key == "azimuths_rad":
            val = value_dict[key]
            tensor_val = val if isinstance(val, torch.Tensor) else torch.tensor(val)
            batch[key] = tensor_val.to(device).repeat(N[0])
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


@bentoml.service(
    name="sv3d-generator",
    image=sv3d_args.image,
    traffic={"timeout": sv3d_args.timeout},
    resources={"gpu": sv3d_args.gpu_count},
)
class SV3DService:
    hf_model = bentoml.models.HuggingFaceModel(
        sv3d_args.model_id, include=[sv3d_args.weights_file]
    )

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_frames = sv3d_args.num_frames

        weights_file_path = os.path.join(self.hf_model, sv3d_args.weights_file)

        self.model = _load_model_architecture(
            sv3d_args.config_file,
            weights_file_path,
            self.device,
            self.num_frames,
            sv3d_args.num_steps,
        )

        if self.device == "cuda":
            self.model = self.model.half()

        self.model.en_and_decode_n_samples_a_time = sv3d_args.decode_chunk_size

    @bentoml.api
    def generate(
        self,
        image: Image.Image,
        elevation_deg: float = sv3d_args.default_elevation,
    ) -> Path:
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        image = image.convert("RGB").resize((576, 576))
        img_tensor = ToTensor()(image) * 2.0 - 1.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device).to(dtype)

        polars_rad = [np.deg2rad(90 - elevation_deg)] * self.num_frames
        azimuths_deg = np.linspace(0, 360, self.num_frames + 1)[1:] % 360
        azimuths_rad = [np.deg2rad((a - azimuths_deg[-1]) % 360) for a in azimuths_deg]

        value_dict = {
            "cond_frames_without_noise": img_tensor,
            "motion_bucket_id": sv3d_args.default_motion_bucket_id,
            "fps_id": sv3d_args.default_fps_id,
            "cond_aug": sv3d_args.default_cond_aug,
            "cond_frames": img_tensor
            + sv3d_args.default_cond_aug * torch.randn_like(img_tensor),
            "polars_rad": torch.tensor(polars_rad, dtype=dtype).to(self.device),
            "azimuths_rad": torch.tensor(azimuths_rad, dtype=dtype).to(self.device),
        }

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(self.device == "cuda")):
                samples_pil = self._sampling(value_dict)

        f = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
        f.close()
        zip_filename = f.name

        # 임시 작업 폴더 생성 (작업 후 자동 삭제됨)
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Creating ZIP file at: {zip_filename}")

            with zipfile.ZipFile(zip_filename, "w") as zipf:
                # 각 프레임 저장 및 압축
                for i, img in enumerate(samples_pil):
                    img_name = f"frame_{i:02d}.png"
                    img_path = os.path.join(temp_dir, img_name)

                    # 이제 파일 잠금 없이 안전하게 저장됩니다.
                    img.save(img_path)
                    zipf.write(img_path, arcname=img_name)

                # GIF 저장 및 압축
                gif_name = "animation.gif"
                gif_path = os.path.join(temp_dir, gif_name)
                samples_pil[0].save(
                    gif_path,
                    save_all=True,
                    append_images=samples_pil[1:],
                    duration=100,
                    loop=0,
                )
                zipf.write(gif_path, arcname=gif_name)

        return Path(zip_filename)

    def _sampling(self, value_dict):
        batch, batch_uc = _get_batch(
            _get_unique_embedder_keys_from_conditioner(self.model.conditioner),
            value_dict,
            [1, self.num_frames],
            T=self.num_frames,
            device=self.device,
        )

        c, uc = self.model.conditioner.get_unconditional_conditioning(
            batch,
            batch_uc=batch_uc,
            force_uc_zero_embeddings=["cond_frames", "cond_frames_without_noise"],
        )

        for k in ["crossattn", "concat"]:
            uc[k] = repeat(uc[k], "b ... -> b t ...", t=self.num_frames)
            uc[k] = rearrange(uc[k], "b t ... -> (b t) ...")
            c[k] = repeat(c[k], "b ... -> b t ...", t=self.num_frames)
            c[k] = rearrange(c[k], "b t ... -> (b t) ...")

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        randn = torch.randn(
            (self.num_frames, 4, 72, 72), device=self.device, dtype=dtype
        )

        additional_model_inputs = {
            "image_only_indicator": torch.zeros(2, self.num_frames).to(self.device),
            "num_video_frames": self.num_frames,
        }

        def denoiser(input, sigma, c):
            return self.model.denoiser(
                self.model.model, input, sigma, c, **additional_model_inputs
            )

        samples_z = self.model.sampler(denoiser, randn, cond=c, uc=uc)

        target_dtype = next(self.model.parameters()).dtype
        samples_z = samples_z.to(target_dtype)

        samples_x = self.model.decode_first_stage(samples_z)
        samples_x = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

        samples_x = rearrange(samples_x, "t c h w -> t h w c").float().cpu().numpy()
        samples_pil = [Image.fromarray((s * 255).astype(np.uint8)) for s in samples_x]

        return samples_pil
