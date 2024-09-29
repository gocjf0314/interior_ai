import torch
import torchvision.transforms
from transformers import pipeline, Pipeline
from PIL import Image


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
    def __call__(self, image: Image, tensor_result: bool = False):
        depth = self.depth_model(image)["depth"]
        depth.save("depth_result.png")
        if tensor_result is True:
            transform = torchvision.transforms.PILToTensor()
            tensor = transform(image)
            return tensor
        else:
            return depth


if __name__ == "__main__":
    pipe = DepthPipeline("cuda")
    result = pipe(image=Image.open("sample_images/room_example.jpg"))

# from transformers import AutoImageProcessor, AutoModelForDepthEstimation
# import torch
# import numpy as np
# from PIL import Image
# import requests
#
# # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# # image = Image.open(requests.get(url, stream=True).raw)
#
# image = Image.open("room_example.jpg")
#
# model_id = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
# image_processor = AutoImageProcessor.from_pretrained(model_id)
# model = AutoModelForDepthEstimation.from_pretrained(model_id)
#
# # prepare image for the model
# inputs = image_processor(images=image, return_tensors="pt")
#
# with torch.no_grad():
#     outputs = model(**inputs)
#     predicted_depth = outputs.predicted_depth
#
# print("Prediction depth size", predicted_depth.size())
#
# # interpolate to original size
# prediction = torch.nn.functional.interpolate(
#     predicted_depth.unsqueeze(1),
#     size=image.size[::-1],
#     mode="bicubic",
#     align_corners=False,
# )[0]
#
# print("Final Prediction deptj size", prediction.size())
#
# import torchvision.transforms as T
#
# vision_transform = T.ToPILImage()
# depth_img = vision_transform(prediction)
# depth_img.save("depth_result.png")
