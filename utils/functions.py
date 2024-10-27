import numpy as np
from typing import Union


def get_image_id(image_path: str):
    image_id = image_path.split('/')[-1]
    image_id = image_id.split('.')[0]
    return image_id


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