import argparse
import os

import cv2
import kornia
import numpy as np
import torch
import torchvision.transforms as T
from diffusers import AutoencoderKL, LCMScheduler, StableDiffusionXLImg2ImgPipeline
from diffusers.utils import export_to_video, load_image
from keyframed.dsl import curve_from_cn_string
from kornia.color import lab_to_rgb, rgb_to_lab
from kornia.geometry.transform import remap
from PIL import Image
from skimage.exposure import match_histograms
from torchvision.models.optical_flow import raft_large
from torchvision.transforms import ToPILImage, ToTensor
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--video_path", type=str)
parser.add_argument("--init_image", type=str)
parser.add_argument("--num_frames", type=int, default=32)
parser.add_argument("--fps", type=int, default=10)
parser.add_argument("--height", type=int, default=1024)
parser.add_argument("--width", type=int, default=1024)
parser.add_argument("--prompt", type=str)
parser.add_argument("--num_inference_steps", type=int, default=12)
parser.add_argument("--strength", type=str, default="0:(0.5)")
parser.add_argument(
    "--model_id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0"
)
parser.add_argument("--lora_id", type=str)
parser.add_argument("--lora_scale", type=float, default=1.0)
parser.add_argument("--save", action="store_true")


def load_video(path, height=512, width=512):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(Image.fromarray(frame).resize((height, width)))

    cap.release()
    return frames


def apply_lab_color_matching(image, reference_image):
    image = ToTensor()(image).unsqueeze(0)
    reference_image = ToTensor()(reference_image).unsqueeze(0)

    image = rgb_to_lab(image)
    reference_image = rgb_to_lab(reference_image)

    output = match_histograms(
        np.array(image[0].permute(1, 2, 0)),
        np.array(reference_image[0].permute(1, 2, 0)),
        channel_axis=-1,
    )

    output = ToTensor()(output).unsqueeze(0)
    output = lab_to_rgb(output)
    output = ToPILImage()(output[0])

    return output


def preprocess(batch):
    transforms = T.Compose(
        [
            T.ToTensor(),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
        ]
    )
    batch = transforms(batch)
    return batch


def get_optical_flow(frames):
    model = raft_large(pretrained=True, progress=False)
    model = model.to(device, torch.float32)
    model = model.eval()

    flow_maps = []
    for frame_1, frame_2 in zip(frames, frames[1:]):
        frame_1 = frame_1.to(device)
        frame_2 = frame_2.to(device)

        with torch.no_grad():
            flow_map = model(frame_1, frame_2)
            flow_maps.append(flow_map[-1].to("cpu"))

        frame_1 = frame_1.to("cpu")
        frame_2 = frame_2.to("cpu")

    model.to("cpu")
    del model
    torch.cuda.empty_cache()

    return flow_maps


def apply_flow_warping(image, flow_map):
    flow_map = flow_map.permute(0, 2, 3, 1)
    image_tensor = ToTensor()(image)

    _, height, width = image_tensor.shape

    meshgrid = kornia.create_meshgrid(height, width, normalized_coordinates=False)
    grid = meshgrid + flow_map
    grid = grid.permute(0, 3, 1, 2)

    output = remap(
        image_tensor[None, :],
        grid[:, 0],
        grid[:, 1],
        mode="bilinear",
        normalized_coordinates=False,
    )
    output_image = ToPILImage()(output[0])
    return output_image


def run(
    save,
    video_path: str,
    init_image: str = None,
    prompt: str = None,
    num_frames: int = 32,
    fps: int = 10,
    num_inference_steps: int = 8,
    height: int = 1024,
    width: int = 1024,
    strength: str = "0:(0.6)",
    model_id: str = None,
    lora_id: str = None,
    lora_scale: float = 1.0,
):
    video_frames = load_video(video_path, height=height, width=width)
    tensor_frames = [preprocess(frame)[None, :] for frame in video_frames[:num_frames]]

    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
    )
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, vae=vae, safety_checker=None
    )
    pipe.set_progress_bar_config(disable=True)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl", adapter_name="lcm")
    if lora_id:
        pipe.load_lora_weights(lora_id, adapter_name="style")
        pipe.set_adapters(["lcm", "style"], adapter_weights=[1.0, lora_scale])

    pipe.enable_model_cpu_offload()

    optical_flow_maps = get_optical_flow(tensor_frames)
    generator = torch.Generator("cpu").manual_seed(42)

    init_image = load_image(args.init_image) if init_image else video_frames[0]

    output = pipe(
        image=init_image,
        prompt=prompt,
        generator=generator,
        num_inference_steps=num_inference_steps,
    ).images[0]
    output_images = [output]
    init_image = output

    strength = curve_from_cn_string(strength)

    pbar = tqdm(total=len(optical_flow_maps), disable=False)
    for frame_idx, flow_map in pbar:
        pbar.set_description(f"Processing {frame_idx}")
        frame = apply_flow_warping(output, flow_map)
        output = pipe(
            image=frame,
            prompt=prompt,
            generator=generator,
            strength=strength[frame_idx],
            num_inference_steps=num_inference_steps,
        ).images[0]

        output = apply_lab_color_matching(output, init_image)
        output_images.append(output)
        if save:
            os.makedirs("generated", exist_ok=True)
            output.save(f"generated/{frame_idx:04d}.png")

    export_to_video(output_images, "output.mp4", fps=fps)

    return


if __name__ == "__main__":
    args = parser.parse_args()
    run(
        args.save,
        args.video_path,
        args.prompt,
        args.init_image,
        args.num_frames,
        args.num_inference_steps,
        args.height,
        args.width,
        args.strength,
        args.model_id,
        args.lora_id,
        args.lora_scale,
    )
