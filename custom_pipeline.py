import torch
import torchvision.transforms
import numpy as np
from PIL import Image
from transformers import (
    AutoImageProcessor,
    UperNetForSemanticSegmentation,
    Pipeline,
    pipeline,
)
from contolnet_interior_design.palette import ade_palette


class RoomSegPipeline:

    def __init__(self):
        # image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
        # image_segmentor = UperNetForSemanticSegmentation.from_pretrained(
        #     "openmmlab/upernet-convnext-small")
        model_id = "openmmlab/upernet-swin-large"
        image_processor = AutoImageProcessor.from_pretrained(model_id)
        image_segmentor = UperNetForSemanticSegmentation.from_pretrained(model_id)
        self.image_processor = image_processor
        self.image_segmentor = image_segmentor

    @torch.inference_mode()
    @torch.autocast('cuda')
    def __call__(self, image: Image) -> Image:
        """Method to segment image
        Args:
            image (Image): input image
        Returns:
            Image: segmented image
        """
        # image_processor, image_segmentor = segmentation.get_segmentation_pipeline()
        pixel_values = self.image_processor(image, return_tensors="pt").pixel_values
        with torch.no_grad():
            outputs = self.image_segmentor(pixel_values)

        seg = self.image_processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]])[0]
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        palette = np.array(ade_palette())
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        color_seg = color_seg.astype(np.uint8)
        seg_image = Image.fromarray(color_seg).convert('RGB')
        return seg_image


class DepthPipeline:
    def __init__(self, device: str):
        self.depth_model = self.load_depth_pipeline(device)

    # load depth pipeline
    def load_depth_pipeline(self, device: str) -> Pipeline:
        depth_pipeline = pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
            device=device
        )
        return depth_pipeline

    @torch.inference_mode()
    @torch.autocast('cuda')
    def __call__(self, image: Image, tensor_result: False) -> Image | torch.Tensor:
        depth = self.depth_model(image)["depth"]
        depth.save("depth_result.png")
        if tensor_result is True:
            transform = torchvision.transforms.PILToTensor()
            tensor = transform(image)
            return tensor
        else:
            return depth