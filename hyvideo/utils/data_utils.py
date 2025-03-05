import numpy as np
import math
from PIL import Image
import torch
import copy
import string
import random

def align_to(value, alignment):
    """align hight, width according to alignment

    Args:
        value (int): height or width
        alignment (int): target alignment factor

    Returns:
        int: the aligned value
    """
    return int(math.ceil(value / alignment) * alignment)

def black_image(width, height):
    black_image = Image.new('RGB', (width, height), (0, 0, 0))
    return black_image

def get_closest_ratio(height: float, width: float, ratios: list, buckets: list):
    aspect_ratio = float(height)/float(width)
    closest_ratio_id = np.abs(ratios - aspect_ratio).argmin()
    closest_ratio = min(ratios, key=lambda ratio: abs(float(ratio) - aspect_ratio))
    return buckets[closest_ratio_id], float(closest_ratio)

def generate_crop_size_list(base_size=256, patch_size=16, max_ratio=4.0):
    num_patches =  round((base_size / patch_size) ** 2)
    assert max_ratio >= 1.
    crop_size_list = []
    wp, hp = num_patches, 1
    while wp > 0:
        if max(wp, hp) / min(wp, hp) <= max_ratio:
            crop_size_list.append((wp * patch_size, hp * patch_size))
        if (hp + 1) * wp <= num_patches:
            hp += 1
        else:
            wp -= 1
    return crop_size_list


def align_floor_to(value, alignment):
    """align hight, width according to alignment

    Args:
        value (int): height or width
        alignment (int): target alignment factor

    Returns:
        int: the aligned value
    """
    return int(math.floor(value / alignment) * alignment)


def augmentation_prompts(structural_prompts):
    # Random drop long caption.
    structural_prompts["long caption"] = random_drop_word(structural_prompts["long caption"])

    # Compose prompts.
    prompts = prompts_compose(structural_prompts)

    return prompts


def random_drop_word(long_caption, text_remove_rate=1.0):
    if text_remove_rate == 0 or random.random() > text_remove_rate:
        return long_caption

    def split_by_unquoted_periods(text):
        result = []
        current_segment = ""
        in_quotes = False

        for char in text:
            if char == '"':
                in_quotes = not in_quotes

            if char == '.' and not in_quotes:
                result.append(current_segment)
                current_segment = ""
            else:
                current_segment += char

        if current_segment:
            result.append(current_segment)

        return result

    def find_ocr_and_drop(text):
        items = split_by_unquoted_periods(text)
        if len(items) == 1:
            return text

        valid_items = []
        for item in items:
            if "\"" in item and any([_x in item.lower() for _x in
                                     ["reads", "reading", "written", "title", "writes", "text", "words",
                                      "chinese characters"]]):
                continue
            valid_items.append(item)
        if len(valid_items) == 0:
            return text
        result = ".".join(valid_items)
        if result and result[-1] not in string.punctuation:
            result += '.'
        return result

    return find_ocr_and_drop(long_caption)


def prompts_compose(structural_prompts):
    long_caption = structural_prompts["long caption"]
    prompts = f"{long_caption}."

    return prompts

def crop_tensor(inputs, crop_width_ratio=1.0, crop_height_ratio=1.0, crop_type="center"):
    b, c, t, h, w = inputs.shape
    crop_h, crop_w = int(h * crop_height_ratio), int(w * crop_width_ratio)

    if crop_type == "center":
        top = (h - crop_h) // 2
        left = (w - crop_w) // 2
    elif crop_type == "random":
        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)

    crop_h = align_floor_to(crop_h, alignment=2)
    crop_w = align_floor_to(crop_w, alignment=2)

    return inputs[:, :, :, top:top + crop_h, left:left + crop_w]

def get_closest_length(length, buckets: list):
    closest_length = buckets[0]
    for b in buckets:
        if length-b>=0: closest_length = b
    return closest_length

def get_temporal_closest_ratio(video_length: float, height: float, width: float, ratios: list, temporal_buckets: list, buckets: list):
    # get closest temporal bucket
    assert video_length >= temporal_buckets[0], "video length is smaller than the first temporal bucket"
    closed_temporal_bucket = get_closest_length(video_length, temporal_buckets)

    # get closest spatial bucket
    aspect_ratio = float(height)/float(width)
    closest_ratio_id = np.abs(ratios - aspect_ratio).argmin()
    closest_ratio = min(ratios, key=lambda ratio: abs(float(ratio) - aspect_ratio))
    return closed_temporal_bucket, buckets[closest_ratio_id], float(closest_ratio)

def generate_crop_size_w_temporal_list(base_size=256, patch_size=16, max_ratio=4.0, temporal_list=[]):
    assert len(temporal_list) > 0, "temporal_list is empty"
    num_patches =  round((base_size / patch_size) ** 2)
    assert max_ratio >= 1.
    crop_size_list = []
    spatial_crop_size_list = []
    wp, hp = num_patches, 1
    while wp > 0:
        if max(wp, hp) / min(wp, hp) <= max_ratio:
            spatial_crop_size_list.append((wp * patch_size, hp * patch_size))
            for temp_bucket in temporal_list:
                crop_size_list.append((int(temp_bucket), wp * patch_size, hp * patch_size))
        if (hp + 1) * wp <= num_patches:
            hp += 1
        else:
            wp -= 1
    return crop_size_list, spatial_crop_size_list