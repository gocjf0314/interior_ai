import torch
from PIL import Image
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation


class OneFormerPipeline:

    def __init__(self, model_id: str | None = None):
        if model_id is None or len(model_id) == 0:
            model_id = "shi-labs/oneformer_ade20k_swin_large"
        self.processor = OneFormerProcessor(model_id)
        self.model = OneFormerForUniversalSegmentation.from_pretrained(model_id)

    def __call__(self, image: Image.Image, task: str = ""):
        # Preprocess image and set inference task
        # task: semantic, instance, panoptic(semantic + instance)
        if len(task) > 0 and task not in ["semantic", "instance", "panoptic"]:
            ValueError(f"Can not use task {task}")

        if len(task) == 0:
            task = "semantic"

        inputs = self.processor(image, [task], return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)

        class_queries_logits = outputs.class_queries_logits
        masks_queries_logits = outputs.masks_queries_logits

        # Post process of output
        predict_result = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0]

        if task == "semantic":
            predicted_map = predict_result
        else:
            predicted_map = predict_result["segmentation"]

    def draw_segmentation(self, segmentation, segments_info):
        pass

    def get_mask(self, segmentation, segments_info):
        pass


if __name__ == "__main__":
    print("")