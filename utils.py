import os
import tempfile

import cv2
import ffmpeg
import numpy as np
import PIL
from kornia.color import lab_to_rgb, rgb_to_lab
from skimage.exposure import match_histograms
from torchvision.transforms import ToPILImage, ToTensor


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


def load_video(path, height=1024, width=1024):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(PIL.Image.fromarray(frame).resize((height, width)))

    cap.release()
    return frames


def export_to_video(images, output_path, fps=10):
    with tempfile.TemporaryDirectory() as temp_dir:
        for i, image in enumerate(images):
            frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
            image.save(frame_path)

        (
            ffmpeg.input(os.path.join(temp_dir, "frame_%04d.png"), framerate=fps)
            .output(output_path, pix_fmt="yuv420p")
            .run(overwrite_output=True)
        )
