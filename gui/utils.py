import numpy as np
import torch
from PIL import ImageTk, Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


class TaskImage:
    def __init__(self, file_path):
        pil_image = Image.open(file_path)
        resized_image = resize_image(pil_image)

        self.original_image = pil_image
        self.photo_image = ImageTk.PhotoImage(resized_image)

        self.segmented_image = None
        self.display_segmented = None
        self.seg_map = None
        self.seg_backup = None


def resize_image(image, max_size=400):
    width, height = image.size
    if width > height:
        new_width = max_size
        new_height = int(max_size * height / width)
    else:
        new_height = max_size
        new_width = int(max_size * width / height)
    resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)
    return resized_image


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_segmenter():
    device = get_device()
    model_name = "matei-dorian/segformer-b5-finetuned-human-parsing"
    feature_extractor = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    model.to(device)
    return model, feature_extractor


def process_seg(seg):
    wanted_labels = {2, 11}
    seg_processed = np.zeros_like(seg)
    seg_processed[np.isin(seg, list(wanted_labels))] = 1
    return seg_processed


def convert_segmap_to_color(x):
    seg_map = np.uint8(x) * 255
    seg_map_rgba = np.zeros((seg_map.shape[0], seg_map.shape[1], 4), dtype=np.uint8)
    seg_map_rgba[..., 0] = 217
    seg_map_rgba[..., 1] = 2
    seg_map_rgba[..., 2] = 125
    seg_map_rgba[..., 3] = seg_map * 0.5
    return seg_map_rgba
