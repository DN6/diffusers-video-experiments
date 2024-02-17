import argparse
import os

import cv2
import kornia
import numpy as np
import torch
import torchvision.transforms as T
from diffusers import (
    AutoencoderKL,
    EulerDiscreteScheduler,
    LCMScheduler,
    ControlNetModel,
    StableDiffusionXLControlNetImg2ImgPipeline,
)
from diffusers.utils import export_to_video, load_image
from keyframed.dsl import curve_from_cn_string
from kornia.color import lab_to_rgb, rgb_to_lab
from kornia.geometry.transform import remap
from PIL import Image
from skimage.exposure import match_histograms
from torchvision.models.optical_flow import raft_large
from torchvision.transforms import ToPILImage, ToTensor
from tqdm import tqdm
from controlnet_aux import MidasDetector, CannyDetector

CONTROLNET_MODELS = ["diffusers/controlnet-canny-sdxl-1.0", "diffusers/controlnet-depth-sdxl-1.0"]

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
parser.add_argument("--guidance_scale", type=float, default=7.5)
parser.add_argument("--canny_scale", type=float, default=0.1)
parser.add_argument("--depth_scale", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--model_id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0"
)
parser.add_argument("--lora_id", type=str)
parser.add_argument("--lora_scale", type=float, default=1.0)
parser.add_argument("--save", action="store_true")
parser.add_argument("--use_lcm", action="store_true")


def load_video(path, height=1024, width=1024):
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


def get_control_images(frames):
    canny = CannyDetector()
    midas = MidasDetector.from_pretrained("lllyasviel/Annotators")
    midas.to(device)

    outputs = []
    for frame in frames:
        processed_image_midas = midas(frame, detect_resolution=1024, image_resolution=1024)
        processed_image_canny = canny(frame, detect_resolution=1024, image_resolution=1024)
        outputs.append([processed_image_canny, processed_image_midas])

    midas.to("cpu")
    del midas
    torch.cuda.empty_cache()

    return outputs


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


def load_controlnet(controlnet_model):
    return ControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.float16)


def run(
    save,
    use_lcm,
    video_path: str,
    init_image: str = None,
    prompt: str = None,
    num_frames: int = 32,
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
    control_images = get_control_images([frame for frame in video_frames[:num_frames]])
    tensor_frames = [preprocess(frame)[None, :] for frame in video_frames[:num_frames]]

    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
    )
    controlnets = [load_controlnet(controlnet_model) for controlnet_model in CONTROLNET_MODELS]

    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        model_id, controlnet=controlnets, vae=vae, torch_dtype=torch.float16, safety_checker=None
    )
    pipe.set_progress_bar_config(disable=True)
    if use_lcm:
        pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl", adapter_name="lcm")
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    else:
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    if lora_id:
        pipe.load_lora_weights(lora_id, adapter_name="style")
        if use_lcm:
            pipe.set_adapters(["lcm", "style"], [1.0, lora_scale])
        else:
            pipe.set_adapters(["style"], [lora_scale])

    pipe.enable_model_cpu_offload()

    optical_flow_maps = get_optical_flow(tensor_frames)
    generator = torch.Generator("cpu").manual_seed(seed)

    init_image = load_image(args.init_image) if init_image else video_frames[0]
    strength = curve_from_cn_string(strength)

    pbar = tqdm(total=len(optical_flow_maps) + 1, disable=False)

    # Generate initial frame
    output = pipe(
        image=init_image,
        control_image=control_images[0],
        prompt=prompt,
        generator=generator,
        strength=strength[0],
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        controlnet_conditioning_scale=[canny_scale],
    ).images[0]
    init_image = output
    output_images = [output]

    pbar.update()

    if save:
        os.makedirs("generated", exist_ok=True)
        output.save("generated/0000.png")

    for flow_idx, flow_map in enumerate(optical_flow_maps):
        frame = apply_flow_warping(output, flow_map)
        frame_idx = flow_idx + 1

        output = pipe(
            image=frame,
            prompt=prompt,
            control_image=control_images[frame_idx],
            controlnet_conditioning_scale=[canny_scale, depth_scale],
            generator=generator,
            strength=strength[frame_idx],
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        ).images[0]

        output = apply_lab_color_matching(output, init_image)
        output_images.append(output)
        if save:
            output.save(f"generated/{frame_idx:04d}.png")
        pbar.update()

    export_to_video(output_images, "output.mp4", fps=fps)

    return


if __name__ == "__main__":
    args = parser.parse_args()
    run(
        args.save,
        args.use_lcm,
        args.video_path,
        args.init_image,
        args.prompt,
        args.num_frames,
        args.fps,
        args.num_inference_steps,
        args.height,
        args.width,
        args.strength,
        args.guidance_scale,
        args.seed,
        args.model_id,
        args.lora_id,
        args.lora_scale,
        args.canny_scale,
        args.depth_scale,
    )
