import sys
# sys.path.append('./')
from PIL import Image
from future.standard_library import import_

from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
from diffusers import DDPMScheduler, AutoencoderKL, ControlNetModel, UNet2DConditionModel
from typing import List

import torch
import os
from transformers import AutoTokenizer
import numpy as np
from utils_mask import get_mask_location
from torchvision import transforms
# from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
from torchvision.transforms.functional import to_pil_image
from room_depth import DepthPipeline
from room_seg import RoomSegPipeline

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Config input values
seed = 42
denoise_steps = 30


def pil_to_binary_mask(pil_image, threshold=0):
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    for i in range(binary_mask.shape[0]):
        for j in range(binary_mask.shape[1]):
            if binary_mask[i, j] == True:
                mask[i, j] = 1
    mask = (mask * 255).astype(np.uint8)
    output_mask = Image.fromarray(mask)
    return output_mask


base_path = 'diffusers/stable-diffusion-xl-1.0-inpainting-0.1'
# example_path = os.path.join(os.path.dirname(__file__), 'example')

unet = UNet2DConditionModel.from_pretrained(
    base_path,
    subfolder="unet",
    torch_dtype=torch.float16,
)
tokenizer_one = AutoTokenizer.from_pretrained(
    base_path,
    subfolder="tokenizer",
    revision=None,
    use_fast=False,
)
tokenizer_two = AutoTokenizer.from_pretrained(
    base_path,
    subfolder="tokenizer_2",
    revision=None,
    use_fast=False,
)
noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")

text_encoder_one = CLIPTextModel.from_pretrained(
    base_path,
    subfolder="text_encoder",
    torch_dtype=torch.float16,
)
text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
    base_path,
    subfolder="text_encoder_2",
    torch_dtype=torch.float16,
)
# image_encoder = CLIPVisionModelWithProjection.from_pretrained(
#     base_path,
#     subfolder="image_encoder",
#     torch_dtype=torch.float16,
# )
vae = AutoencoderKL.from_pretrained(
    base_path,
    subfolder="vae",
    torch_dtype=torch.float16,
)

# "stabilityai/stable-diffusion-xl-base-1.0",
UNet_Encoder = UNet2DConditionModel.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    subfolder="unet",
    torch_dtype=torch.float16,
)
# UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
#     base_path,
#     subfolder="unet_encoder",
#     torch_dtype=torch.float16,
# )

# parsing_model = Parsing(0)
# openpose_model = OpenPose(0)

UNet_Encoder.requires_grad_(False)
# image_encoder.requires_grad_(False)
vae.requires_grad_(False)
unet.requires_grad_(False)
text_encoder_one.requires_grad_(False)
text_encoder_two.requires_grad_(False)
tensor_transfrom = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

pipe = TryonPipeline.from_pretrained(
    base_path,
    unet=unet,
    vae=vae,
    feature_extractor=CLIPImageProcessor(),
    text_encoder=text_encoder_one,
    text_encoder_2=text_encoder_two,
    tokenizer=tokenizer_one,
    tokenizer_2=tokenizer_two,
    scheduler=noise_scheduler,
    # image_encoder=image_encoder,
    torch_dtype=torch.float16,
)
pipe.unet_encoder = UNet_Encoder

# Load Segmentation Controlnet
segment_pipeline = RoomSegPipeline()

# Load Depth ControlNet
depth_pipeline = DepthPipeline(device=device)

garment_des = ""

# openpose_model.preprocessor.body_estimation.model.to(device)
pipe.to(device)
pipe.unet_encoder.to(device)


# args = apply_net.create_argument_parser().parse_args(('show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml',
#                                                       './ckpt/densepose/model_final_162be9.pkl', 'dp_segm', '-v',
#                                                       '--opts', 'MODEL.DEVICE', 'cuda'))
# # verbosity = getattr(args, "verbosity", None)
# depth_img = args.func(args, human_img_arg)
# depth_img = depth_img[:, :, ::-1]
# depth_img = Image.fromarray(depth_img).resize((768, 1024))

def run():
    garm_img = Image.open("sample_images/garm_img.png")  # PIL.Image, Style image recommended by user needs
    garm_img = garm_img.convert("RGB").resize((768, 1024))
    room_img_origin = Image.open("sample_images/room_example.jpg").convert("RGB")
    depth_img = depth_pipeline(room_img_origin)

    # Crop image size
    width, height = room_img_origin.size
    target_width = int(min(width, height * (3 / 4)))
    target_height = int(min(height, width * (4 / 3)))

    left = (width - target_width) / 2
    top = (height - target_height) / 2
    right = (width + target_width) / 2
    bottom = (height + target_height) / 2

    cropped_img = room_img_origin.crop((left, top, right, bottom))
    crop_size = cropped_img.size
    room_img = cropped_img.resize((768, 1024))

    # mask, _ = get_mask_location('hd', "upper_body", model_parse, keypoints)
    mask = segment_pipeline(image=room_img, get_mask=True)
    mask = mask.resize((768, 1024))

    mask_gray = (1 - transforms.ToTensor()(mask)) * tensor_transfrom(room_img)
    mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)

    room_seg, mask = segment_pipeline(room_img, get_mask=True)

    with torch.cuda.amp.autocast():
        with torch.no_grad():
            prompt = "model is wearing " + garment_des
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
            with torch.inference_mode():
                # prompt 전처리
                (
                    prompt_embeds,
                    negative_prompt_embeds,
                    pooled_prompt_embeds,
                    negative_pooled_prompt_embeds,
                ) = pipe.encode_prompt(
                    prompt,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                    negative_prompt=negative_prompt,
                )

                prompt = "a photo of " + garment_des
                negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                if not isinstance(prompt, List):
                    prompt = [prompt] * 1
                if not isinstance(negative_prompt, List):
                    negative_prompt = [negative_prompt] * 1
                with torch.inference_mode():
                    (
                        prompt_embeds_c,
                        _,
                        _,
                        _,
                    ) = pipe.encode_prompt(
                        prompt,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=False,
                        negative_prompt=negative_prompt,
                    )

                depth_img = tensor_transfrom(Image.open(depth_img)).unsqueeze(0).to(device, torch.float16)
                garm_tensor = tensor_transfrom(garm_img).unsqueeze(0).to(device, torch.float16)
                generator = torch.Generator(device).manual_seed(seed) if seed is not None else None
                images = pipe(
                    prompt_embeds=prompt_embeds.to(device, torch.float16),
                    negative_prompt_embeds=negative_prompt_embeds.to(device, torch.float16),
                    pooled_prompt_embeds=pooled_prompt_embeds.to(device, torch.float16),
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device, torch.float16),
                    num_inference_steps=denoise_steps,
                    generator=generator,
                    strength=1.0,
                    pose_img=depth_img.to(device, torch.float16),
                    text_embeds_cloth=prompt_embeds_c.to(device, torch.float16),
                    cloth=garm_tensor.to(device, torch.float16),
                    mask_image=mask,
                    image=room_img,
                    height=1024,
                    width=768,
                    ip_adapter_image=garm_img.resize((768, 1024)),
                    guidance_scale=2.0,
                )[0]


    out_img = images[0].resize(crop_size)
    room_img_origin.paste(out_img, (int(left), int(top)))
    room_img_origin.save("img_save_path")
