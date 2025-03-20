from typing import List, Dict, Any
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer
from internvl.model.internvl_chat import InternVLWithHiMTok


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    # print(f'width: {width}, height: {height}, best_ratio: {best_ratio}')
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


class Predictor:
    def __init__(self, model_path, max_num=1,sam=None):
        self.max_num = max_num
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        self.model = InternVLWithHiMTok.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True)

        self.model.mask_decoder.init_tt_ids(self.tokenizer)
        if sam:
            self.model.mask_decoder.load_sam(model_path=sam)
        self.model.eval().cuda()

    def update_metrics(self, mask_images, batch_samples, trackers=None):
        ious = []
        box_iou_scores = []
        for data, mask_image in zip(batch_samples, mask_images):
            gt_mask = data["mask"]
            #mask_image is numpy array, resize to gt's long side, then clip the padding
            mask_image = Image.fromarray(mask_image).resize((gt_mask.shape[1], gt_mask.shape[0]), Image.NEAREST)
            mask_image = np.array(mask_image)
            # mask_image = mask_image[:gt_mask.shape[0], :gt_mask.shape[1]]
            mask_image[gt_mask == 255] = 1
            intersection, union, iou = self.compute_iou(gt_mask, mask_image)
            ious.append(iou)
            if trackers is not None:
                trackers['intersection'].update(intersection, n=1)
                trackers['union'].update(union, n=1)
                trackers['gIoU'].update(iou, n=1)
        return ious

    def compute_iou(self, gt_mask, mask_image):
        gt_mask = gt_mask > 0.5
        mask_image = mask_image > 0.5
        intersection = np.logical_and(gt_mask, mask_image)
        union = np.logical_or(gt_mask, mask_image)
        intersection = np.sum(intersection)
        union = np.sum(union)
        iou = intersection / (union + 1e-10)
        if union == 0:
            iou = 1.0
        return intersection, union, iou

    def predict(self, batch_samples: List[Dict[str, Any]],return_response: bool = False):

        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)

        def build_transform(input_size):
            MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
            transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD)
            ])
            return transform

        transform = build_transform(input_size=448)

        generation_config = dict(
            max_new_tokens=128, 
            do_sample=False
        )

        def load_image(image):
            images = dynamic_preprocess(image, min_num=1, max_num=self.max_num, image_size=448, use_thumbnail=True)
            pixel_values = [transform(image) for image in images]
            pixel_values = torch.stack(pixel_values)
            return pixel_values

        pixel_values = [load_image(data["image"]) for data in batch_samples]
        num_patches_list = [pixel_values[i].size(0) for i in range(len(pixel_values))]
        pixel_values = torch.cat(pixel_values, dim=0)
        pixel_values = pixel_values.to(self.model.dtype).to(self.model.device)
        
        questions = [data["prompt"] for data in batch_samples]

        responses, mask_images = self.model.batch_chat(self.tokenizer, pixel_values,
                                    num_patches_list=num_patches_list,
                                    questions=questions,
                                    generation_config=generation_config,
                                    decode_mask=True)

        if self.model.mask_decoder.sam is not None:
            images = []
            for data in batch_samples:
                image = data["image"].resize((1024, 1024))
                image = torch.tensor(np.array(image))
                images.append(image)
            images = torch.stack(images, dim=0).to(self.model.device)
            images = images.permute(0, 3, 1, 2)
            mask_images = self.model.mask_decoder.decode_mask_sam(mask_images.float().unsqueeze(1), images).mean(dim=1, keepdim=False)

        return mask_images.detach()
