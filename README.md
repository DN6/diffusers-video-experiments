# Video Experiments

This repo is a collection of scripts to experiment with generating videos using [Diffusers](https://github.com/huggingface/diffusers/tree/main)

Currently it only has scripts for hybrid video generation, but more scripts will be added in the future.

## Hybrid Video

This is a simple implementation of [Deforum](https://deforum.github.io/)'s hybrid video feature.

### How to use

#### Install dependencies

```shell
pip install -r requirements.txt
```

#### Run Inference Script

The hybrid video script uses an [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) with the option of using [LCM-LoRA](https://huggingface.co/latent-consistency/lcm-lora-sdxl) for fast inference. It requires a source video and a text prompt.

Run the script with the default settings using the following command:

```shell
python hybrid_video.py --video_path <path_to_video> \
--prompt "<your prompt>"
```

`--video_path` is the file path to the video you want to use to drive the motion of your animation. `--prompt` is the text prompt you want to use to generate the content of the video.

You can run the script with the example source video to verify results

```shell
python hybrid_video.py --video_path assets/waves.mp4 \
--prompt "a painting in the style of Van Gogh"
```

#### Using an Initial Image

Hybrid video can use an initial image as a starting point for the generation process. You can pass in the path to an image using the `--init_image` argument. This can either be a URL or a file path to an image

```shell
python hybrid_video.py --video_path <path_to_video> \
--prompt "<your prompt>" \
--init-image "<url_or_path_to_image>"
```

#### Setting Number of Frames

You can configure how many frames of the video to process using the `--num_frames` argument.

```shell
python hybrid_video.py --video_path <path_to_video> \
--prompt "<your prompt>" \
--num_frames <number_of_frames>
```

#### Varying the denoising strength

The hybrid video script uses the same type of schedule format found in Deforum. This allows you to vary the strength and guidance scale of the denoising process over time. You can set the denoising strength schedule using the `--strength` argument.

For example the following command will set the denoising strength to 0.75 at frame 0 and then drop it down to 0.5 from frame 1 onwards.

```shell
python hybrid_video.py --video_path <path_to_video> \
--prompt "<your prompt>" \
--strength "0:(0.75), 1:(0.75)"
```

This following command sets the strength to 0.75 throughout the generation process.

```shell
python hybrid_video.py --video_path <path_to_video> \
--prompt "<your prompt>" \
--strength "0:(0.75)"
```

#### Setting the Seed

You can set the seed for the random number generator using the `--seed` argument.

```shell
python hybrid_video.py --video_path <path_to_video> \
--prompt "<your prompt>" \
--seed 12345
```

#### Setting the Guidance Scale

You can set the guidance scale using the `--guidance_scale` argument.

```shell
python hybrid_video.py --video_path <path_to_video> \
--prompt "<your prompt>" \
--guidance_scale 9.0
```

#### Using LoRAs

You can load any SDXL compatible LoRA model using the `--lora_id` argument, and passing in either a LoRA repo id on the Hugging Face hub or a local path to a directory with a LoRA model.

[LoRA Studio](https://huggingface.co/spaces/enzostvs/lora-studio) has a number of models that you can use with this script.

Set the LoRA scale using the `--lora_scale` argument.

```shell
python hybrid_video.py --video_path <path_to_video> \
--prompt "<your prompt>" \
--lora_id "<lora-repo-id>" \
--lora_scale 1.0
```


#### Saving intermediate results

You can save the intermediate framss from the generation process using the the `--save` argument

The frames will be saved to a directory named `generated` in the current working directory.

```shell
python hybrid_video.py --video_path <path_to_video> \
--prompt "<your prompt>" \
--save
```

#### Using ControlNets.

The `hybrid_video_controlnet.py` uses the exact same arguments as `hybrid_video.py` with two addtional parameters, `--canny_scale` and `--depth_scale` that set the conditioning scales for a Canny and Depth ControlNet model.

```shell
python hybrid_video_controlnet.py --video_path <path_to_video> \
--prompt "<your prompt>" \
--canny_scale 0.5 \
--dpeth_scale 0.1
```