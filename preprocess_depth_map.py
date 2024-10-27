import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image


# 이미지 불러오기 및 전처리 함수 정의
def load_image(image_path, size=(256, 256)):
    """이미지를 로드하고 정규화된 텐서로 변환합니다."""
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    size = h, w
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)  # 배치 차원 추가


def load_depth_map(depth_path, size=(256, 256)):
    """깊이 맵을 로드하고 정규화합니다."""
    depth = Image.open(depth_path).convert("L")  # 흑백 이미지로 변환
    w, h = depth.size
    size = h, w
    transform = transforms.Compose([
        transforms.Resize(size),
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


# 전처리 함수
def preprocess_image(original_image, depth_map):
    """깊이 맵과 원본 이미지를 활용한 전처리."""
    confidence_map = create_confidence_map(depth_map)  # 신뢰도 맵 생성
    filtered_depth = onion_peel_filtering(depth_map, confidence_map)  # 양파 껍질 필터링 적용

    # 깊이 맵을 사용해 비트레이트 감소를 위해 픽셀 선택
    masked_image = original_image * confidence_map

    return masked_image, filtered_depth


# 예시 사용
if __name__ == "__main__":
    # 이미지와 깊이 맵 로드
    original_image = load_image("data/sample_images/room_example.jpg")
    depth_map = load_depth_map("data/sample_images/depth_result.png")

    # 전처리 수행
    processed_image, processed_depth = preprocess_image(original_image, depth_map)

    # 결과 출력 (확인용)
    print("Processed Image Shape:", processed_image.shape)
    print("Processed Depth Map Shape:", processed_depth.shape)

    # PIL로 변환해 저장 (원본 형태로 확인)
    processed_image_pil = transforms.ToPILImage()(processed_image.squeeze(0))
    processed_image_pil.save("processed_image.jpg")

    processed_depth_pil = transforms.ToPILImage()(processed_depth.squeeze(0))
    processed_depth_pil.save("processed_depth.png")
