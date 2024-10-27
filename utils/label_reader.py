import os
import json

DATASET_ROOT = "C:/data/interior_ai_data"


def _get_json_path(filename: str):
    return os.path.join(DATASET_ROOT, filename)


def get_json_file(json_path: str) -> dict:
    with open(json_path, "r") as fp:
        json_data = dict(json.load(fp))
        return json_data


def get_category_keys(data: dict):
    return list(data.keys())


class InteriorLabelLeader:
    def __init__(self):
        label_path = os.path.join(DATASET_ROOT, "data_labels.json")
        labels = get_json_file(label_path)
        self.object_type = labels["object_type"]
        self.object_style = labels["object_style"]
        self.object_texture = labels["object_texture"]
        self.object_material = labels["object_material"]
        self.room_type = labels["room_type"]
        self.room_style = labels["room_style"]

        train_anno_path = os.path.join(DATASET_ROOT, "train", "train_annotation.json")
        self.train_annotation = list(get_json_file(train_anno_path)["annotations"])

    def get_object_types(self):
        return get_category_keys(self.object_type)

    def get_object_styles(self):
        return get_category_keys(self.object_style)

    def get_object_textures(self):
        return get_category_keys(self.object_texture)

    def get_object_materials(self):
        return get_category_keys(self.object_material)

    def get_room_types(self):
        return get_category_keys(self.room_type)

    def get_room_styles(self):
        return get_category_keys(self.room_style)

    def get_object_all_tags(self):
        tag_list = []
        tag_list.extend(self.get_object_types())
        tag_list.extend(self.get_object_styles())
        tag_list.extend(self.get_object_materials())
        tag_list.extend(self.get_object_textures())
        return tag_list

    def get_room_all_tags(self):
        tag_list = []
        tag_list.extend(self.get_room_styles())
        tag_list.extend(self.get_room_styles())
        return tag_list
