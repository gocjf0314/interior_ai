import torch
import numpy as np
from PIL import Image
from diffusers import ControlNetModel
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
from contolnet_interior_design.colors import ade_palette, COLOR_MAPPING_


class RoomSegPipeline:

    def __init__(self):
        # image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
        # image_segmentor = UperNetForSemanticSegmentation.from_pretrained(
        #     "openmmlab/upernet-convnext-small")
        model_id = "openmmlab/upernet-swin-large"  # Use swin transformer large as backbone of upernet
        image_processor = AutoImageProcessor.from_pretrained(model_id)
        image_segmentor = UperNetForSemanticSegmentation.from_pretrained(model_id)
        self.image_processor = image_processor
        self.image_segmentor = image_segmentor

    @torch.inference_mode()
    @torch.autocast('cuda')
    def __call__(self, image: Image, get_mask: bool = False, seg_label: int | None = None):
        """Method to segment image
        Args:
            image (Image): input image
        Returns:
            Image: segmented image
        """
        pixel_values = self.image_processor(image, return_tensors="pt").pixel_values
        with torch.no_grad():
            outputs = self.image_segmentor(pixel_values)

        seg = self.image_processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]])[0]
        seg_height, seg_width = seg.shape[:2]
        color_seg = np.zeros((seg_height, seg_width, 3), dtype=np.uint8)

        palette = np.array(ade_palette())
        if seg_label is None:
            for label, color in enumerate(palette):
                color_seg[seg == label, :] = color
            color_seg = color_seg.astype(np.uint8)
            seg_image = Image.fromarray(color_seg).convert('RGB')
        else:
            seg[seg != seg_label] = -1
            for label, color in enumerate(palette):
                color_seg[seg == label, :] = color
            color_seg = color_seg.astype(np.uint8)
            seg_image = Image.fromarray(color_seg).convert('RGB')

            if get_mask is True:
                mask_img = np.array(color_seg.copy(), dtype=np.uint8)
                mask = (mask_img != [0, 0, 0]).any(axis=-1)
                mask_img[mask] = [255, 255, 255]
                mask_img = Image.fromarray(mask_img).convert("RGB")
                return seg_image, mask_img

            return seg_image, None

        return seg_image, None


if __name__ == "__main__":
    from transformers import SwinConfig, UperNetConfig, UperNetForSemanticSegmentation

    backbone_config = SwinConfig(out_features=["stage1", "stage2", "stage3", "stage4"])

    config = UperNetConfig(backbone_config=backbone_config)
    model = UperNetForSemanticSegmentation(config)

    model_id = "openmmlab/upernet-swin-large"  # Use swin transformer large as backbone of upernet
    image_processor = AutoImageProcessor.from_pretrained(model_id)

    result = model(
        Image.open("sample_images/room_example.jpg").load(),
        labels=[v for k, v in COLOR_MAPPING_.items()]
    )
    print(type(result))
    print(result.size)
    # pipe = RoomSegPipeline()
    # image = Image.open("room_example.jpg")
    # seg_color, mask = pipe(image=image, get_mask=False)
    # seg_color.save("seg_result.png")
    # if mask is not None:
    #     mask.save("seg_mask.png")
