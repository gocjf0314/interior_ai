import torch
import numpy as np
from PIL import Image
from transformers import (
    UperNetForSemanticSegmentation,
    SegformerImageProcessor,
)
from contolnet_interior_design.colors import ade_palette, COLOR_MAPPING_


class UperNetPipeline:

    def __init__(self):
        _model_id = "openmmlab/upernet-swin-large"
        # _model_id = "openmmlab/upernet-convnext-xlarge"
        self.image_processor = SegformerImageProcessor.from_pretrained(_model_id)
        self.model = UperNetForSemanticSegmentation.from_pretrained(_model_id)

    def __call__(self, image: Image, seg_label: int | None = None):
        """Method to segment image
        Args:
            image (Image): input image
        Returns:
            Image: segmented image
        """
        return self.predict(image=image, seg_label=seg_label)

    def predict(self, image: Image, seg_label: int | None = None):
        inputs = self.image_processor(image, return_tensors="pt")

        # 픽셀 값 텐서로 추출한 뒤 인퍼런스
        with torch.no_grad():
            outputs = self.model(**inputs)
            # output logits shape: [1, 150, 512, 512]

        seg = self.image_processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=[image.size[::-1]]
        )[0]  # shape: [w, h], 각 요소들은 라벨 넘버

        # seg 배열은 [height, width] 크기의 라벨 넘버 배열
        seg_height, seg_width = seg.shape[:2]

        # color_seg 배열을 RGB 3채널로 초기화
        color_seg = np.zeros((seg_height, seg_width, 3), dtype=np.uint8)

        # 흑백 마스크용 배열 초기화
        mask_img = np.zeros((seg_height, seg_width, 3), dtype=np.uint8)

        # ADE20K 팔레트 (또는 사용하고 있는 팔레트) 불러오기
        palette = np.array(ade_palette())  # shape: [num_classes, 3]
        seg_color_set = set(seg)
        # print(seg_color_set)

        # 각 라벨에 맞는 색상을 color_seg 배열에 할당
        if seg_label is None:
            for label, color in enumerate(palette):
                color_seg[seg == label] = color  # 해당 라벨 위치에 색상 할당

                # 마스크 생성: 해당 라벨에 해당하는 영역을 흰색(255, 255, 255)으로 설정
                # label 0 : background
                mask_img[seg != 0] = [255, 255, 255]
        else:
            for label, color in enumerate(palette):
                color_seg[seg == label] = color  # 해당 라벨 위치에 색상 할당

                # 마스크 생성: 해당 라벨에 해당하는 영역을 흰색(255, 255, 255)으로 설정
                # label 0 : background
                mask_img[seg == seg_label] = [255, 255, 255]

        # mask_img를 RGB 흑백 이미지로 변환 (모든 값이 0 또는 255)
        mask_img = Image.fromarray(mask_img).convert("RGB")

        # color_seg 이미지를 RGB로 변환하여 시각화
        color_seg = color_seg.astype(np.uint8)
        seg_image = Image.fromarray(color_seg).convert('RGB')
        return seg_image, mask_img


if __name__ == "__main__":
    pipe = UperNetPipeline()
    room_img = Image.open("../data/sample_images/room_example.jpg")
    seg_info = "lamp"
    seg_info = list(COLOR_MAPPING_.values()).index(seg_info) - 1
    print(seg_info)
    # seg_color, mask = pipe(image=room_img, seg_label=seg_info)
    seg_color, mask = pipe(image=room_img)
    seg_color.save("seg_result.png")
    mask.save("seg_mask.png")
