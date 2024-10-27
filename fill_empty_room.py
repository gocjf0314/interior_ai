import cv2
import torch
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from diffusers import (
    StableDiffusionXLControlNetInpaintPipeline,
    ControlNetModel,
)

from src.depth_anythingv2_pipeline import DepthPipeline
from src.upernet_pipeline import UperNetPipeline
from utils.functions import filter_items
from contolnet_interior_design.colors import COLOR_MAPPING_, map_colors_rgb, to_rgb

device = "cuda" if torch.cuda.is_available() else "cpu"

positive_affection = ", interior design, 4K, high resolution, photorealistic, majestic, realistic"
negative_affection = ", lowres, watermark, banner, logo, watermark, contactinfo, text, deformed, blurry, blur, out of focus, out of frame, surreal, ugly"

depth_controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-depth-sdxl-1.0",
    use_safetensors=True,
)

pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
    pretrained_model_name_or_path='diffusers/stable-diffusion-xl-1.0-inpainting-0.1',
    controlnet=depth_controlnet,
    use_safetensors=True,
).to(device=device)
print("!!Load SDXL Inpainting Model!!\n")

pipe.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="sdxl_models",
    image_encoder_folder="image_encoder",
    weight_name="ip-adapter-plus_sdxl_vit-h.safetensors"
)
print("!!Load IP Adapter!!\n")
pipe.enable_model_cpu_offload()


def decorate_side_sections(room_prompt: str, empty_room_image: Image.Image, style_image: Image.Image):
    """
    변, 천장, 바닥을 전체 마스킹
    방의 전체적인 스타일/느낌 적용
    """
    # Process segmentation and Generate mask
    segment_pipeline = UperNetPipeline()
    seg_color, mask = segment_pipeline(image=empty_room_image.copy())
    seg_color.save("room_seg_color.png")

    segmentation_mask = np.array(seg_color.copy(), dtype=np.uint8)
    # unique_colors = np.unique(segmentation_mask.reshape(-1, segmentation_mask.shape[2]), axis=0)
    # unique_colors = [tuple(color) for color in unique_colors]
    # segment_items = [map_colors_rgb(i) for i in unique_colors]

    control_items = ["background", "wall", "ceiling"]
    color_map_invert = {v: to_rgb(k) for k, v in COLOR_MAPPING_.items()}
    control_colors = [color_map_invert[item] for item in control_items]

    chosen_colors, segment_items = filter_items(
        colors_list=control_colors,
        items_list=control_items,
        items_to_remove=[]
    )

    mask = np.zeros_like(segmentation_mask)
    for color in chosen_colors:
        color_matches = (segmentation_mask == color).all(axis=2)
        mask[color_matches] = 1

    # enlarge mask region so that it also will erase the neighborhood of masked stuff
    mask_image = Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB")
    # mask_image = mask_image.filter(ImageFilter.MaxFilter(25))
    mask_image.save("mask_image.png")

    # Process depth-map
    depth_pipeline = DepthPipeline(device=device)
    controlnet_image = depth_pipeline(empty_room_image)
    controlnet_image = depth_pipeline.tensor_to_pil_image(controlnet_image)
    controlnet_image.save("depth_image.png")
    print("!!Pre-Processed All images!!\n")

    seed = 2024
    strength = 0.8
    guidance_scale = 15
    controlnet_conditioning_scale = 0.5
    room_prompt += positive_affection
    # negative_prompt = "window, door, furniture, objects, low resolution, banner, logo, watermark, text, deformed, blurry, out of focus, surreal, ugly, beginner"
    negative_prompt = "Additional objects, furniture, structural changes, decorations, windows, doors, or floor modifications."
    result = pipe(
        prompt=room_prompt,
        negative_prompt=negative_prompt,
        image=empty_room_image.convert("RGB"),
        mask_image=mask_image.convert("RGB"),
        control_image=controlnet_image.convert("RGB"),
        ip_adapter_image=style_image.convert("RGB"),
        generator=torch.Generator(device=device).manual_seed(seed),
        strength=strength,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
    ).images[0]
    print("!!Finish Inference!!\n")

    return result


def decorate_side_objects(room_prompt: str, room_image: Image.Image, garment_image: Image.Image | None = None):
    pass


if __name__ == "__main__":
    image = Image.open("interior_test2.png").convert("RGB")
    room_style = Image.open("data/sample_images/garm_img.png")
    prompt = "Empty room, mask only walls and ceiling. Apply [desired style, e.g., 'modern industrial'], keeping room empty without added objects, furniture, or structural changes. Retain floor and original layout."
    decorated_room_sections = decorate_side_sections(
        room_prompt=prompt,
        empty_room_image=image,
        style_image=room_style
    )
    temp = np.array(decorated_room_sections, dtype=np.uint8)
    cv2.imwrite("deco_empty_result.png", temp[:, :, ::-1])
    # cv2.waitKey(0)
