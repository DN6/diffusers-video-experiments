import argparse
import json
import os
from datetime import datetime

import kornia
import torch
from controlnet_aux import CannyDetector, ZoeDetector
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DPMSolverSinglestepScheduler,
    StableDiffusionXLControlNetImg2ImgPipeline,
)
from diffusers.utils import load_image, load_video, export_to_video
from keyframed.dsl import curve_from_cn_string
from kornia.geometry.transform import remap
from torchvision.transforms import ToPILImage, ToTensor
from tqdm import tqdm

from processors import OpticalFlowProcessor
from utils import apply_lab_color_matching
from wonderwords import RandomWord

GEN_OUTPUT_PATH = os.getenv("GEN_OUTPUT_PATH", "generated_hybrid_videos")
CONTROLNET_MODELS = [
    "xinsir/controlnet-canny-sdxl-1.0",
    "xinsir/controlnet-depth-sdxl-1.0",
]

negative_prompt = "worst quality, low quality"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--video_path", type=str)
parser.add_argument("--init_image", type=str)
parser.add_argument("--num_frames", type=int, default=32)
parser.add_argument("--cadence", type=int, default=1)
parser.add_argument("--fps", type=int, default=10)
parser.add_argument("--height", type=int, default=1024)
parser.add_argument("--width", type=int, default=1024)
parser.add_argument("--prompt", type=str)
parser.add_argument("--num_inference_steps", type=int, default=12)
parser.add_argument("--strength", type=str, default="0:(0.5)")
parser.add_argument("--guidance_scale", type=float, default=7.5)
parser.add_argument("--canny_scale", type=float, default=0.1)
parser.add_argument("--depth_scale", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
parser.add_argument("--lora_id", type=str)
parser.add_argument("--lora_scale", type=float, default=1.0)
parser.add_argument("--save", action="store_true")

canny = CannyDetector()
depth_detect = ZoeDetector.from_pretrained("lllyasviel/Annotators")
depth_detect.to(device)


def get_control_images(frames):
    outputs = []
    for frame in frames:
        processed_image_midas = depth_detect(frame, detect_resolution=1024, image_resolution=1024)
        processed_image_canny = canny(frame, detect_resolution=1024, image_resolution=1024)
        outputs.append([processed_image_canny, processed_image_midas])

    return outputs


def apply_loopback_controlnet(frame):
    processed_image_canny = canny(frame, detect_resolution=1024, image_resolution=1024)
    processed_image_midas = depth_detect(frame, detect_resolution=1024, image_resolution=1024)

    return [processed_image_canny, processed_image_midas]


def apply_flow_warping(image, flow_map):
    flow_map = flow_map.permute(0, 2, 3, 1)
    image_tensor = ToTensor()(image)

    _, height, width = image_tensor.shape

    meshgrid = kornia.create_meshgrid(height, width, normalized_coordinates=False)
    grid = meshgrid - (flow_map * 1.0)
    grid = grid.permute(0, 3, 1, 2)

    output = remap(
        image_tensor[None, :],
        grid[:, 0],
        grid[:, 1],
        mode="bilinear",
        normalized_coordinates=False,
        padding_mode="border",
    )
    output_image = ToPILImage()(output[0])
    return output_image


def load_controlnet(controlnet_model):
    return ControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.float16)


def run(
    save,
    save_path,
    video_path: str,
    init_image: str = None,
    prompt: str = None,
    num_frames: int = 32,
    cadence: int = 1,
    fps: int = 10,
    num_inference_steps: int = 16,
    height: int = 1024,
    width: int = 1024,
    strength: str = "0:(0.6)",
    guidance_scale: float = 7.5,
    seed: int = 42,
    model_id: str = None,
    lora_id: str = None,
    lora_scale: float = 1.0,
    canny_scale: float = 0.0,
    depth_scale: float = 0.0,
):
    video_frames = load_video(video_path, height=height, width=width)
    video_frames = [
        video_frames[frame_idx] for frame_idx in range(0, min(len(video_frames), num_frames * cadence), cadence)
    ]
    optical_flow_maps = OpticalFlowProcessor()(video_frames, device)

    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    controlnets = [load_controlnet(controlnet_model) for controlnet_model in CONTROLNET_MODELS]

    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        model_id,
        controlnet=controlnets,
        vae=vae,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config)

    if lora_id:
        pipe.load_lora_weights(lora_id, adapter_name="style")
        pipe.set_adapters(["style"], [lora_scale])

    pipe.set_progress_bar_config(disable=True)

    control_images = get_control_images(video_frames)
    generator = torch.Generator("cpu").manual_seed(seed)

    init_image = load_image(args.init_image).resize((height, width)) if init_image else video_frames[0]
    strength = curve_from_cn_string(strength)

    pbar = tqdm(total=len(optical_flow_maps) + 1, disable=False)

    pipe.to("cuda")
    # Generate initial frame
    output = pipe(
        image=init_image,
        control_image=control_images[0],
        prompt=prompt,
        negative_prompt=negative_prompt,
        generator=generator,
        strength=strength[0],
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        controlnet_conditioning_scale=[canny_scale, depth_scale],
    ).images[0]

    pbar.update()
    pipe.to("cpu")

    init_image = output
    ip_adapter_image = init_image
    output_images = [output]

    if save:
        output.save(f"{save_path}/0000.png")
    output.save(f"{save_path}/preview.png")

    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
    scale = {
        "down": {"block_2": [0.0, 1.0]},
        "up": {"block_0": [0.0, 1.0, 0.0]},
    }
    pipe.to("cuda")
    pipe.set_ip_adapter_scale(scale)

    control_image = apply_loopback_controlnet(output)

    for flow_idx, flow_map in enumerate(optical_flow_maps):
        frame = apply_flow_warping(output, flow_map)
        frame_idx = flow_idx + 1

        output = pipe(
            image=frame,
            prompt=prompt,
            negative_prompt=negative_prompt,
            control_image=control_image,
            controlnet_conditioning_scale=[canny_scale, depth_scale],
            strength=strength[frame_idx],
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            ip_adapter_image=[ip_adapter_image],
        ).images[0]

        output = apply_lab_color_matching(output, init_image)
        ip_adapter_image = output
        control_image = apply_loopback_controlnet(output)

        output_images.append(output)
        if save:
            output.save(f"{save_path}/{frame_idx:04d}.png")
        output.save(f"{save_path}/preview.png")

        pbar.update()

    export_to_video(output_images, f"{save_path}/output.mp4", fps=fps)


if __name__ == "__main__":
    args = parser.parse_args()
    config = vars(args)

    wordgen = RandomWord()
    run_name = (
        f"{wordgen.word(include_parts_of_speech=['adjectives'])}-{wordgen.word(include_parts_of_speech=['nouns'])}"
    )
    timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M")
    run_id = f"hv-controlnet-{timestamp}-{run_name}"
    save_path = f"{GEN_OUTPUT_PATH}/{run_id}"
    os.makedirs(save_path, exist_ok=True)

    with open(f"{save_path}/config.json", "w") as fp:
        json.dump(config, fp, indent=4)

    config.update({"save_path": save_path})
    run(**config)
