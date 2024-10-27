from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from PIL import Image
import requests


# url = "https://huggingface.co/datasets/shi-labs/oneformer_demo/blob/main/ade20k.jpeg"
# image = Image.open(requests.get(url, stream=True).raw)

image = Image.open("data/sample_images/room_example.jpg")


# Loading a single model for all three tasks
processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_dinat_large")
model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_dinat_large")

# Semantic Segmentation
semantic_inputs = processor(images=image, task_inputs=["semantic"], return_tensors="pt")
semantic_outputs = model(**semantic_inputs)
# pass through image_processor for postprocessing
predicted_semantic_map = processor.post_process_semantic_segmentation(semantic_outputs, target_sizes=[image.size[::-1]])[0]

# Instance Segmentation
instance_inputs = processor(images=image, task_inputs=["instance"], return_tensors="pt")
instance_outputs = model(**instance_inputs)
# pass through image_processor for postprocessing
predicted_instance_map = processor.post_process_instance_segmentation(instance_outputs, target_sizes=[image.size[::-1]])[0]["segmentation"]

# Panoptic Segmentation
panoptic_inputs = processor(images=image, task_inputs=["panoptic"], return_tensors="pt")
panoptic_outputs = model(**panoptic_inputs)
# pass through image_processor for postprocessing
predicted_panoptic_map = processor.post_process_panoptic_segmentation(panoptic_outputs, target_sizes=[image.size[::-1]])[0]["segmentation"]
