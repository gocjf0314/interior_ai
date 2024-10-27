import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import DPTImageProcessor, DepthAnythingForDepthEstimation


def load_image(image_path):
    """이미지를 로드하고 정규화된 텐서로 변환합니다."""
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)  # 배치 차원 추가


def load_depth_map(depth_path):
    """깊이 맵을 로드하고 정규화된 텐서로 반환합니다."""
    depth = Image.open(depth_path).convert("L")  # 흑백 이미지로 변환
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform(depth).unsqueeze(0)  # 배치 차원 추가


# 신뢰도 맵 생성
def create_confidence_map(depth, threshold=0.1):
    """깊이 맵의 신뢰도를 계산합니다."""
    confidence_map = (depth > threshold).float()  # 신뢰할 수 있는 영역 표시
    return confidence_map


# 양파 껍질 필터링 구현
def onion_peel_filtering(depth, confidence, iterations=5):
    """외부에서 내부로 순차적으로 채워넣는 필터링."""
    filtered_depth = depth.clone()
    for _ in range(iterations):
        # 인접 픽셀의 평균으로 빈 영역 채우기
        mask = (confidence == 0).float()
        avg_neighbors = F.avg_pool2d(filtered_depth * mask, kernel_size=3, stride=1, padding=1)
        filtered_depth = torch.where(mask > 0, avg_neighbors, filtered_depth)
    return filtered_depth


class DepthPipeline:
    def __init__(self, model_id: str | None = None, device: str = "cpu"):
        # self.depth_model = self.load_depth_pipeline(device)
        _model_id = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
        if model_id is not None and len(model_id) > 0:
            _model_id = model_id

        # DPT(Dense Prediction Transformer) Image Processor
        self.image_processor = DPTImageProcessor.from_pretrained(_model_id)
        self.depth_model = DepthAnythingForDepthEstimation.from_pretrained(_model_id)

    def predict(self, image: Image.Image):
        # prepare image for the model
        processed = self.image_processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.depth_model(**processed)
            predicted_depth = outputs.predicted_depth

        # interpolate to original size
        prediction: torch.Tensor = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),  # 두번째 차원에 1차원 추가 하여 [batch_size, 1, height, width] 형태가 됨
            size=image.size[::-1],  # h, w -> w, h
            mode="bicubic",  # 2D에서 3차 보간, linear / bilinear: 1D 또는 2D 선형 보간
            align_corners=False,  # 모서리 픽셀 기준 정렬
        )  # [batch_size, 1, h, w]

        return prediction

    @staticmethod
    def tensor_to_pil_image(tensor) -> Image.Image:
        output = tensor.squeeze().cpu().numpy()

        # shape: (h, w)
        # dtype: np.uint8
        formatted = (output * 255 / np.max(output)).astype("uint8")

        # result image shape(numpy 기준): (h, w, 3)
        img = Image.fromarray(formatted)
        return img

    @staticmethod
    def preprocess(self, origin_image, depth_map):
        """깊이 맵과 원본 이미지를 활용한 전처리."""
        confidence_map = create_confidence_map(depth_map)  # 신뢰도 맵 생성
        filtered_depth = onion_peel_filtering(depth_map, confidence_map)  # 양파 껍질 필터링 적용

        # 깊이 맵을 사용해 비트레이트 감소를 위해 픽셀 선택
        masked_image = origin_image * confidence_map
        return filtered_depth, masked_image

    def __call__(self, image: Image.Image):
        return self.predict(image)

if __name__ == "__main__":
    pipe = DepthPipeline()

    img_path = "../data/dataset_sample/room1.png"
    input_img = Image.open(img_path)
    result = pipe(image=input_img)  # return torch.Tensor

    result = pipe.tensor_to_pil_image(result)  # return PIL.Image
    print("Final Prediction depth size", result.size)
    result.save("depth_result.png")

    # Load images as torch tensor
    original_image = load_image(img_path)
    depth_img = load_depth_map("dataset_sample/depth_result.png")

    processed_depth, mask = pipe.preprocess(
        original_image,
        depth_img
    )

    processed_image_pil = transforms.ToPILImage()(mask.squeeze(0))
    processed_image_pil.save("processed_image.jpg")

    processed_depth_pil = transforms.ToPILImage()(processed_depth.squeeze(0))
    processed_depth_pil.save("processed_depth.png")

    cv2_img = cv2.imread("dataset_sample/depth_result.png")
    print(cv2_img.shape)
