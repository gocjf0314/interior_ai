from typing import Union
import random
import cv2
import torch
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from diffusers import (
    StableDiffusionXLControlNetInpaintPipeline,
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    AutoencoderKL,
)

from src.depth_anythingv2_pipeline import DepthPipeline
from src.upernet_pipeline import UperNetPipeline
from contolnet_interior_design.colors import COLOR_MAPPING_, map_colors_rgb


device = "cuda" if torch.cuda.is_available() else "cpu"

positive_affection = ", interior design, 4K, high resolution, photorealistic, majestic, realistic"
negative_affection = ", lowres, watermark, banner, logo, watermark, contactinfo, text, deformed, blurry, blur, out of focus, out of frame, surreal, ugly"


def inpaint_object(room_image: Image.Image, object_style_image: Image.Image, object_label: str, object_caption: str):
    # ellljoy/controlnet-interior-design
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        torch_dtype=torch.float16
    )
    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0",
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
    )
    pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        pretrained_model_name_or_path='diffusers/stable-diffusion-xl-1.0-inpainting-0.1',
        controlnet=controlnet,
        vae=vae,
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
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

    # Process segmentation and Generate mask
    segment_pipeline = UperNetPipeline()
    # object_seg = "sofa;couch;lounge"
    object_label = list(COLOR_MAPPING_.values()).index(object_label) - 1
    _, mask = segment_pipeline(image=room_image, seg_label=object_label)
    # mask.save("mask.png")
    # mask = transforms.pil_to_tensor(mask).to(device=device, dtype=torch.float16)

    # Process depth-map
    depth_pipeline = DepthPipeline(device=device)
    depth_img = depth_pipeline(room_image)
    depth_img = depth_pipeline.tensor_to_pil_image(depth_img)
    # depth_img.save("depth.png")

    # prompt = "A room contain a lighting, floor lamp, modern style, fabric material, smooth texture, beige color"
    # object_caption = "armchair, loveseat, modern style, fabric material, smooth texture, gray color"
    prompt = f"A photo of {object_caption} in the room"

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_affection,
        image=room_image,
        control_image=depth_img.convert("RGB"),
        ip_adapter_image=object_style_image,
        generator=torch.Generator(device=device).manual_seed(42),
        num_inference_steps=30,
        controlnet_conditioning_scale=0.5,
    ).images[0]
    print("!!Finish Inference!!\n")

    return result


def decorate_empty_room(empty_room_img: Image.Image, ip_adapter_img: Image.Image, prompt: str):
    # vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0",
        # variant="fp16",
        use_safetensors=True,
        # torch_dtype=torch.float16,
    )
    # pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    #     pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0",
    #     controlnet=controlnet,
    #     # vae=vae,
    #     # variant="fp16",
    #     use_safetensors=True,
    #     # torch_dtype=torch.float16,
    # ).to(device=device)
    pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        pretrained_model_name_or_path='diffusers/stable-diffusion-xl-1.0-inpainting-0.1',
        controlnet=controlnet,
        # vae=vae,
        # variant="fp16",
        use_safetensors=True,
        # torch_dtype=torch.float16,
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

    # Process segmentation and Generate mask
    segment_pipeline = UperNetPipeline()
    seg_color, mask = segment_pipeline(image=empty_room_img.copy())
    # mask.save("mask.png")

    segmentation_mask = np.array(seg_color.copy())
    unique_colors = np.unique(segmentation_mask.reshape(-1, segmentation_mask.shape[2]), axis=0)
    unique_colors = [tuple(color) for color in unique_colors]
    segment_items = [map_colors_rgb(i) for i in unique_colors]

    control_items = ["windowpane;window", "sconce", "door;double;door", "light;light;source", "stairs;steps",
                     "escalator;moving;staircase;moving;stairway"]
    chosen_colors, segment_items = filter_items(
        colors_list=unique_colors,
        items_list=control_items,
        items_to_remove=segment_items
    )

    mask = np.zeros_like(segmentation_mask)
    for color in chosen_colors:
        color_matches = (segmentation_mask == color).all(axis=2)
        mask[color_matches] = 1

    mask_image = Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB")
    # enlarge mask region so that it also will erase the neighborhood of masked stuff
    mask_image = mask_image.filter(ImageFilter.MaxFilter(25))
    mask_image = ImageOps.invert(mask_image.convert("L"))
    mask_image = mask_image.convert("RGB")
    mask_image.save("mask_image.png")

    # Process depth-map
    depth_pipeline = DepthPipeline(device=device)
    depth_img = depth_pipeline(empty_room_img)
    depth_img = depth_pipeline.tensor_to_pil_image(depth_img)
    depth_img.save("depth.png")
    print("!!Processed All images!!\n")

    prompt += positive_affection

    negative_prompt = "window, door, low resolution, banner, logo, watermark, text, deformed, blurry, out of focus, surreal, ugly, beginner"

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=empty_room_img,
        mask_image=mask_image,
        control_image=depth_img,
        ip_adapter_image=ip_adapter_img,
        generator=torch.Generator(device=device).manual_seed(42),
        strength=0.7,
        # num_inference_steps=25,
        guidance_scale=10,
        controlnet_conditioning_scale=0.8,
    ).images[0]
    print("!!Finish Inference!!\n")

    return result


def filter_items(
    colors_list: Union[list, np.ndarray],
    items_list: Union[list, np.ndarray],
    items_to_remove: Union[list, np.ndarray]
):
    filtered_colors = []
    filtered_items = []
    for color, item in zip(colors_list, items_list):
        if item not in items_to_remove:
            filtered_colors.append(color)
            filtered_items.append(item)
    return filtered_colors, filtered_items


def get_empty_room(room_image: Image):
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    # controlnet = ControlNetModel.from_pretrained(
    #     "diffusers/controlnet-depth-sdxl-1.0",
    #     variant="fp16",
    #     use_safetensors=True,
    #     torch_dtype=torch.float16,
    # )

    controlnet = ControlNetModel.from_pretrained(
        "SargeZT/sdxl-controlnet-seg",
        # variant="fp16",
        torch_dtype=torch.float16,
    )

    pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        pretrained_model_name_or_path='diffusers/stable-diffusion-xl-1.0-inpainting-0.1',
        controlnet=controlnet,
        vae=vae,
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
    ).to(device=device)
    # pipe.load_ip_adapter(
    #     "h94/IP-Adapter",
    #     subfolder="sdxl_models",
    #     image_encoder_folder="image_encoder",
    #     weight_name="ip-adapter-plus_sdxl_vit-h.safetensors"
    # )
    pipe.enable_model_cpu_offload()
    print("!!Load SDXL Inpainting Model!!\n")

    # Process segmentation and Generate mask
    segment_pipeline = UperNetPipeline()
    # seg_color, mask = segment_pipeline(image=room_image.copy())
    # mask.save("mask.png")
    seg_color, mask = segment_pipeline(image=room_image)
    # unique_labels = np.unique(np.array(seg_label))
    # seg_labels = list(COLOR_MAPPING_.values())

    segmentation_mask = np.array(seg_color.copy())
    unique_colors = np.unique(segmentation_mask.reshape(-1, segmentation_mask.shape[2]), axis=0)
    unique_colors = [tuple(color) for color in unique_colors]
    segment_items = [map_colors_rgb(i) for i in unique_colors]

    control_items = ["windowpane;window", "wall", "floor;flooring", "ceiling", "sconce", "door;double;door",
                     "light;light;source", "painting;picture", "stairs;steps",
                     "escalator;moving;staircase;moving;stairway"]
    chosen_colors, segment_items = filter_items(
        colors_list=unique_colors,
        items_list=segment_items,
        items_to_remove=control_items
    )

    mask = np.zeros_like(segmentation_mask)
    for color in chosen_colors:
        color_matches = (segmentation_mask == color).all(axis=2)
        mask[color_matches] = 1

    # enlarge mask region so that it also will erase the neighborhood of masked stuff
    mask_image = Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB")
    mask_image = mask_image.filter(ImageFilter.MaxFilter(25))
    mask_image.save("mask.png")

    controlnet_image = Image.fromarray(segmentation_mask.astype(np.uint8)).convert("RGB")
    # controlnet_image = segmentation_mask.filter(ImageFilter.MaxFilter(25))
    controlnet_image.save("control_image.png")

    # # Process depth-map
    # depth_pipeline = DepthPipeline(device=device)
    # depth_img = depth_pipeline(room_image)
    # depth_img = depth_pipeline.tensor_to_PILImage(depth_img)
    print("!!Process segmentation!!\n")

    inpaint_prompt = "Empty room, with only empty walls, floor, ceiling, door, windows"
    inpaint_negative = ("furnitures, sofa, cough, table, plants, rug, home equipment, music equipment, shelves, "
                        "books, light, lamps, window, door, radiator")

    seed = random.randint(0, 2147483646)
    seed = 10
    print(f"seed: {seed}")

    result = pipe(
        prompt="Completely Empty room, empty walls, empty floor, empty ceiling, no furnitures or objects",
        negative_prompt=inpaint_negative,
        image=room_image,
        mask_image=mask_image,
        control_image=controlnet_image,
        # ip_adapter_image=Image.open("empty_room_garm.png").convert("RGB"),
        strength=0.8,
        guidance_scale=20.0,
        controlnet_conditioning_scale=0.5,
        generator=torch.Generator(device=device).manual_seed(seed),
    ).images[0]
    result = result.convert("RGB")
    return result


if __name__ == "__main__":
    # input_room_img = Image.open("data/sample_images/room_example.jpg").convert("RGB")
    #
    # empty_room = get_empty_room(input_room_img)
    # empty_room.save("empty_room.png")

    empty_room_path = "data/sample_images/empty_room.png"
    empty_room = Image.open(empty_room_path).convert("RGB")

    prompt = "Modern minimalist bedroom with matte gray built-in cabinets, a light wooden desk, brown chair, light gray and beige bedding, herringbone wood floor, large window with white blinds, soft natural light, and simple decor on the shelves."
    decorated_room = decorate_empty_room(
        empty_room_img=empty_room,
        ip_adapter_img=Image.open("data/sample_images/garm_img.png").convert("RGB"),
        prompt=prompt,
    )

    save_path = "data/test_outputs/decorated_room.png"
    decorated_room.save(save_path)

    # room_style = Image.open("sample_images/garm_img.png")
    # filled_room = decorate_empty_room(empty_room, room_style)
    # filled_room.save("filled_room.png")
