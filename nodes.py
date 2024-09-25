import os
import torch
import hashlib
import numpy as np
from PIL import Image

import folder_paths
import comfy.model_management as mm

from huggingface_hub import snapshot_download

from diffusers import (
    DDPMScheduler,
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    StableDiffusionXLControlNetPipeline
)
from diffusers.utils import load_image
from diffusers.models.controlnet_sd3 import SD3ControlNetModel
from .pipeline_stable_diffusion_3_controlnet_inpainting import StableDiffusion3ControlNetInpaintingPipeline
from .controlnet_flux import FluxControlNetModel
from .transformer_flux import FluxTransformer2DModel
from .pipeline_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline


def convert_preview_image(images):
    # 转换图像为 torch.Tensor，并调整维度顺序为 NHWC
    images_tensors = []
    for img in images:
        # 将 PIL.Image 转换为 numpy.ndarray
        img_array = np.array(img)
        # 转换 numpy.ndarray 为 torch.Tensor
        img_tensor = torch.from_numpy(img_array).float() / 255.
        # 转换图像格式为 CHW (如果需要)
        if img_tensor.ndim == 3 and img_tensor.shape[-1] == 3:
            img_tensor = img_tensor.permute(2, 0, 1)
        # 添加批次维度并转换为 NHWC
        img_tensor = img_tensor.unsqueeze(0).permute(0, 2, 3, 1)
        images_tensors.append(img_tensor)

    if len(images_tensors) > 1:
        output_image = torch.cat(images_tensors, dim=0)
    else:
        output_image = images_tensors[0]
    return output_image


def make_inpaint_condition(init_image, mask_image):
    init_image = np.array(init_image.convert("RGB")).astype(np.float32) / 255.0
    mask_image = np.array(mask_image.convert("L")).astype(np.float32) / 255.0
    assert init_image.shape[0:1] == mask_image.shape[0:1], "image and image_mask must have the same image size"
    init_image[mask_image > 0.5] = -1.0  # set as masked pixel
    init_image = np.expand_dims(init_image, 0).transpose(0, 3, 1, 2)
    init_image = torch.from_numpy(init_image)
    return init_image


def add_fg(full_img, fg_img, mask_img):
    full_img = np.array(full_img).astype(np.float32)
    fg_img = np.array(fg_img).astype(np.float32)
    mask_img = np.array(mask_img).astype(np.float32) / 255.
    full_img = full_img * mask_img + fg_img * (1 - mask_img)
    return Image.fromarray(np.clip(full_img, 0, 255).astype(np.uint8))


def download(model, model_dir):
    if not os.path.exists(model_dir):
        print(f"Downloading {model}")
        snapshot_download(repo_id=model, local_dir=model_dir, local_dir_use_symlinks=False)
        # huggingface-cli download --resume-download --local-dir-use-symlinks False LinkSoul/LLaSM-Cllama2 --local-dir LLaSM-Cllama2


class EcomXL_LoadImage:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"image": (sorted(files), {"image_upload": True})},
                }

    CATEGORY = "AliControlnetInpainting"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "load_image"

    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)

        output_image = load_image(image_path)
        return (output_image,)


class EcomXL_Controlnet_ModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    [
                        "alimama-creative/EcomXL_controlnet_inpaint",
                    ],
                    {"default": "alimama-creative/EcomXL_controlnet_inpaint"}),
            },
            "optional": {
                # "variant": ("STRING", {"default": "fp16"}),
                "use_safetensors": ("BOOLEAN", {"default": False})
            }
        }

    RETURN_TYPES = ("CONTROLNET",)
    RETURN_NAMES = ("controlnet",)
    FUNCTION = "load_controlnet"
    CATEGORY = "AliControlnetInpainting"

    def load_controlnet(self, model, use_safetensors):
        device = mm.get_torch_device()
        # import pdb;pdb.set_trace()
        model_name = model.rsplit('/', 1)[-1]
        model_dir = (os.path.join(folder_paths.models_dir, "inpaint", model_name))
        download(model, model_dir)
        controlnet = ControlNetModel.from_pretrained(
            model_dir,
            # variant=variant,
            use_safetensors=use_safetensors).to(device)
        # controlnet.enable_model_cpu_offload()
        return (controlnet,)


class EcomXL_SDXL_Inpaint_ModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    ["stabilityai/stable-diffusion-xl-base-1.0"],
                    {"default": "stabilityai/stable-diffusion-xl-base-1.0"},
                ),
                "controlnet": ("CONTROLNET",)
            },
            "optional": {
                # "variant": ("STRING", {"default": "fp16"}),
                "use_safetensors": ("BOOLEAN", {"default": False})
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "AliControlnetInpainting"

    def load_model(self, model, controlnet, use_safetensors):
        device = mm.get_torch_device()
        model_name = model.rsplit('/', 1)[-1]
        model_dir = (os.path.join(folder_paths.models_dir, "checkpoints", model_name))
        download(model, model_dir)
        pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
            model,
            controlnet=controlnet,
            # variant=variant,
            use_safetensors=use_safetensors).to(device)
        pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
        return (pipeline,)


class EcomXL_Condition:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "condition"
    CATEGORY = "AliControlnetInpainting"

    def condition(self, image, mask):
        mask = Image.fromarray(255 - np.array(mask))
        control_image = make_inpaint_condition(image, mask)
        return (control_image, mask)


class EcomXL_AddFG:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "inpainting_image": ("IMAGE",),
                "image": ("IMAGE",),
                "mask": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "add_fg"
    CATEGORY = "AliControlnetInpainting"

    def add_fg(self, inpainting_image, image, mask):
        images = add_fg(inpainting_image, image, mask)
        output_images = convert_preview_image([images])
        return (output_images,)


class SD3_Controlnet_ModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    [
                        "alimama-creative/SD3-Controlnet-Inpainting",
                    ],
                    {"default": "alimama-creative/SD3-Controlnet-Inpainting"}),
            },
            "optional": {
                # "variant": ("STRING", {"default": "fp16"}),
                "use_safetensors": ("BOOLEAN", {"default": False})
            }
        }

    RETURN_TYPES = ("CONTROLNET",)
    RETURN_NAMES = ("controlnet",)
    FUNCTION = "load_controlnet"
    CATEGORY = "AliControlnetInpainting"

    def load_controlnet(self, model, use_safetensors):
        device = mm.get_torch_device()
        model_name = model.rsplit('/', 1)[-1]
        model_dir = (os.path.join(folder_paths.models_dir, "inpaint", model_name))
        download(model, model_dir)

        controlnet = SD3ControlNetModel.from_pretrained(
            model_dir,
            # variant=variant,
            use_safetensors=use_safetensors, extra_conditioning_channels=1,
            low_cpu_mem_usage=False, ignore_mismatched_sizes=True
        ).to(device)
        # controlnet.enable_model_cpu_offload()
        return (controlnet,)


class SD3_Inpainting_ModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    ["stabilityai/stable-diffusion-3-medium-diffusers"],
                    {"default": "stabilityai/stable-diffusion-3-medium-diffusers"},
                ),
                "controlnet": ("CONTROLNET",)
            },
            "optional": {
                # "variant": ("STRING", {"default": "fp16"}),
                "use_safetensors": ("BOOLEAN", {"default": False})
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "AliControlnetInpainting"

    def load_model(self, model, controlnet, use_safetensors):
        device = mm.get_torch_device()
        model_name = model.rsplit('/', 1)[-1]
        model_dir = (os.path.join(folder_paths.models_dir, "checkpoints", model_name))
        download(model, model_dir)

        pipeline = StableDiffusion3ControlNetInpaintingPipeline.from_pretrained(
            model_dir,
            controlnet=controlnet,
            # variant=variant,
            use_safetensors=use_safetensors,
            torch_dtype=torch.float16,
        )
        pipeline.text_encoder.to(torch.float16)
        pipeline.controlnet.to(torch.float16)
        pipeline.to(device)
        return (pipeline,)


class Flux_Controlnet_ModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    [
                        "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha",
                    ],
                    {"default": "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha"}),
            },
            "optional": {
                # "variant": ("STRING", {"default": "fp16"}),
                "use_safetensors": ("BOOLEAN", {"default": False})
            }
        }

    RETURN_TYPES = ("CONTROLNET",)
    RETURN_NAMES = ("controlnet",)
    FUNCTION = "load_controlnet"
    CATEGORY = "AliControlnetInpainting"

    def load_controlnet(self, model, use_safetensors):
        device = mm.get_torch_device()
        model_name = model.rsplit('/', 1)[-1]
        model_dir = (os.path.join(folder_paths.models_dir, "inpaint", model_name))
        download(model, model_dir)
        controlnet = FluxControlNetModel.from_pretrained(
            model_dir,
            # variant=variant,
            use_safetensors=use_safetensors).to(device)
        # controlnet.enable_model_cpu_offload()
        return (controlnet,)


class Flux_Inpainting_ModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    ["black-forest-labs/FLUX.1-dev"],
                    {"default": "black-forest-labs/FLUX.1-dev"},
                ),
                "controlnet": ("CONTROLNET",)
            },
            "optional": {
                # "variant": ("STRING", {"default": "fp16"}),
                "use_safetensors": ("BOOLEAN", {"default": False})
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "AliControlnetInpainting"

    def load_model(self, model, controlnet, use_safetensors):
        device = mm.get_torch_device()
        model_name = model.rsplit('/', 1)[-1]
        model_dir = (os.path.join(folder_paths.models_dir, "checkpoints", model_name))
        download(model, model_dir)

        transformer = FluxTransformer2DModel.from_pretrained(
            model_dir, subfolder='transformer', torch_dytpe=torch.bfloat16
        )
        pipeline = FluxControlNetInpaintingPipeline.from_pretrained(
            model_dir,
            controlnet=controlnet,
            transformer=transformer,
            torch_dtype=torch.bfloat16
        ).to(device)
        pipeline.transformer.to(torch.bfloat16)
        pipeline.controlnet.to(torch.bfloat16)
        return (pipeline,)


class AliInpaintingsampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
                "negative_prompt": ("STRING", {"multiline": True,
                                               "default": "deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, NSFW", }),
                "width": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 8}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 500, "step": 1}),
                "guidance_scale": ("FLOAT", {"default": 7.0, "min": 1.0, "max": 100.0, "step": 0.5}),
                "controlnet_conditioning_scale": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                "true_guidance_scale": ("FLOAT", {"default": 3.5, "min": 1.0, "max": 100.0, "step": 0.5}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                # "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "mode": (['ecomxl', 'sd3', 'flux.1'], {"default": 'ecomxl'})
            },
            "optional": {
                "mask": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "inpaint"
    CATEGORY = "AliControlnetInpainting"

    def inpaint(self,
                model,
                image,
                prompt,
                negative_prompt,
                width,
                height,
                steps,
                guidance_scale,
                controlnet_conditioning_scale,
                true_guidance_scale,
                seed,
                # batch_size,
                mode,
                mask=None):
        device = mm.get_torch_device()

        generator = torch.Generator(device).manual_seed(seed)
        if mode == "ecomxl":
            images = model(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                generator=generator,
                # batch_size=batch_size,
            ).images[0]
        if mode == "sd3":
            images = model(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                control_image=image,
                control_mask=mask,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                # true_guidance_scale=true_guidance_scale,
                generator=generator,
                # batch_size=batch_size,
            ).images[0]
            images = convert_preview_image(images)
        if mode == "flux.1":
            images = model(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                control_image=image,
                control_mask=mask,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                generator=generator,
                true_guidance_scale=true_guidance_scale
            ).images[0]
            images = convert_preview_image(images)
        return (images,)


NODE_CLASS_MAPPINGS = {
    "EcomXL_LoadImage": EcomXL_LoadImage,
    "EcomXL_Controlnet_ModelLoader": EcomXL_Controlnet_ModelLoader,
    "EcomXL_SDXL_Inpaint_ModelLoader": EcomXL_SDXL_Inpaint_ModelLoader,
    "EcomXL_Condition": EcomXL_Condition,
    "EcomXL_AddFG": EcomXL_AddFG,
    "SD3_Controlnet_ModelLoader": SD3_Controlnet_ModelLoader,
    "SD3_Inpainting_ModelLoader": SD3_Inpainting_ModelLoader,
    "Flux_Controlnet_ModelLoader": Flux_Controlnet_ModelLoader,
    "Flux_Inpainting_ModelLoader": Flux_Inpainting_ModelLoader,
    "AliInpaintingsampler": AliInpaintingsampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EcomXL_LoadImage": "EcomXL LoadImage",
    "EcomXL_Controlnet_ModelLoader": "EcomXL Controlnet Model Loader",
    "EcomXL_SDXL_Inpaint_ModelLoader": "EcomXL SDXL Inpaint Model Loader",
    "EcomXL_Condition": "EcomXL Condition",
    "EcomXL_AddFG": "EcomXL Add FG",
    "SD3_Controlnet_ModelLoader": "SD3 Controlnet Model Loader",
    "SD3_Inpainting_ModelLoader": "SD3 Inpainting Model Loader",
    "Flux_Controlnet_ModelLoader": "Flux Controlnet Model Loader",
    "Flux_Inpainting_ModelLoader": "Flux Inpainting Model Loader",
    "AliInpaintingsampler": "Ali Inpainting Sampler"
}
