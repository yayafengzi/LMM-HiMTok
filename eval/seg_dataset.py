import os
import json
import numpy as np
import random
from torch.utils.data import Dataset
from pycocotools import mask
from PIL import Image, ImageOps
from glob import glob
import cv2

class ReferSegDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        image_dir,
        refer_seg_data="refcoco||refcoco+||refcocog||grefcoco",
        split="val",
        text_mode='random'
    ):
        self.image_dir = image_dir
        self.text_mode = text_mode
        self.data = json.load(open(os.path.join(dataset_dir, f"{refer_seg_data}/{refer_seg_data}_{split}.json"), "r"))
        if refer_seg_data == "grefcoco":
            self.data = [item for item in self.data if len(item['instruction'])>0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = self.data[i]
        image_info = item['image_info']
        m_final = np.zeros(
                        (image_info["height"], image_info["width"])
                    ).astype(np.uint8)
        instruction = item['instruction']
        sentences = [instruction[j]['sent'] for j in range(len(instruction))]
        if len(sentences) == 0:
            sentence = ''
        else:
            if self.text_mode == 'random':
                sentence = random.choice(sentences)
            elif self.text_mode == 'first':
                sentence = sentences[0]
            elif self.text_mode == 'all':    
                sentence = "Meet all the descriptions: "
                for i, sent in enumerate(sentences):
                    sentence += f"{i+1}. {sent}. "
            for ann in item['anns']:
                if len(ann["segmentation"]) == 0:
                    m = np.zeros(
                        (image_info["height"], image_info["width"])
                    ).astype(np.uint8)
                else:
                    if type(ann["segmentation"]) == list:  # polygon
                        rle = mask.frPyObjects(
                            ann["segmentation"], image_info["height"], image_info["width"], )
                    else:
                        rle = ann["segmentation"]
                        # 处理counts为列表的情况
                        if isinstance(rle["counts"], list):
                            # 将counts列表转换为bytes格式
                            rle = mask.frPyObjects(
                                [rle], image_info["height"], image_info["width"]
                            )
                        elif not isinstance(rle["counts"], bytes):
                            rle["counts"] = rle["counts"].encode()
                    m = mask.decode(rle)
                    m = np.sum(
                        m, axis=2
                    )  # sometimes there are multiple binary map (corresponding to multiple segs)
                    m = m.astype(np.uint8)  # convert to np.uint8
                m_final = m_final | m

        image_path = os.path.join(self.image_dir, image_info['file_name'])
        image = Image.open(image_path).convert("RGB")
        prompt = "Segment <ref>{}</ref>.".format(sentence)

        data_dict = {
            "image":image,
            "mask":m_final,
            "prompt":prompt
        }
        return data_dict

class ReasonSegDataset(Dataset):
    def __init__(
        self,
        dataset_dir="/data1/chensenda/datas/reasonseg/ReasonSeg",
        split="val",
        max_num=-1,
        text_mode='random'
    ):
        self.text_mode = text_mode
        self.data = glob(os.path.join(dataset_dir, f"{split}/*.json"))

    def __len__(self):
        return len(self.data)

    def get_mask_from_json(self, json_path, size):
        try:
            with open(json_path, "r") as r:
                anno = json.loads(r.read())
        except:
            with open(json_path, "r", encoding="cp1252") as r:
                anno = json.loads(r.read())

        inform = anno["shapes"]
        comments = anno["text"]
        is_sentence = anno["is_sentence"]

        height, width = size

        ### sort polies by area
        area_list = []
        valid_poly_list = []
        for i in inform:
            label_id = i["label"]
            points = i["points"]
            if "flag" == label_id.lower():  ## meaningless deprecated annotations
                continue

            tmp_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.polylines(tmp_mask, np.array([points], dtype=np.int32), True, 1, 1)
            cv2.fillPoly(tmp_mask, np.array([points], dtype=np.int32), 1)
            tmp_area = tmp_mask.sum()

            area_list.append(tmp_area)
            valid_poly_list.append(i)

        ### ground-truth mask
        sort_index = np.argsort(area_list)[::-1].astype(np.int32)
        sort_index = list(sort_index)
        sort_inform = []
        for s_idx in sort_index:
            sort_inform.append(valid_poly_list[s_idx])

        ret_mask = np.zeros((height, width), dtype=np.uint8)
        for i in sort_inform:
            label_id = i["label"]
            points = i["points"]

            if "ignore" in label_id.lower():
                label_value = 255  # ignored during evaluation
            else:
                label_value = 1  # target

            cv2.polylines(ret_mask, np.array([points], dtype=np.int32), True, label_value, 1)
            cv2.fillPoly(ret_mask, np.array([points], dtype=np.int32), label_value)

        return ret_mask, comments, is_sentence

    def __getitem__(self, i):
        json_path = self.data[i]
        image_path = json_path.replace('.json', '.jpg')
        image = Image.open(image_path).convert("RGB")
        image = ImageOps.exif_transpose(image)
        ret_mask, sentences, is_sentence = self.get_mask_from_json(json_path, (image.height, image.width))

        if self.text_mode == 'random':
            sentence = random.choice(sentences)
        elif self.text_mode == 'first':
            sentence = sentences[0]
        elif self.text_mode == 'all':
            sentence = ','.join(sentences)
        if is_sentence:
            prompt = sentence + "\nWhat is the object and mask in the image?"
        else:
            prompt = "What is " + sentence + "?\nWhat is the object and mask in the image?"

        data_dict = {
            "image":image,
            "mask":ret_mask,
            "prompt":prompt
        }
        return data_dict
