import argparse
import itertools
import json
import os
import random
import re
import time
from functools import partial

import torch
from internvl.model import load_model_and_tokenizer
from internvl.train.dataset import build_transform, dynamic_preprocess
from PIL import Image
from torchvision.ops.boxes import box_area
from tqdm import tqdm
from typing import Union, List

class Scorer:
    def __init__(self):
        self.scores = []
    
    def clear(self):
        self._init()
    
    def average(self, clear=False):
        res = sum(self.scores) / len(self.scores) if len(self.scores) > 0 else -1.0
        return res

    def get_a_score(
        self,
        pred:str,
        gt:Union[str,List[str]],
        metric:str,
        word_processor=None,
        question=None,
    ) -> float:

        if metric == "choice_acc":
            func = self.mcq_acc
        score = func(gt, pred)
        self.scores.append(score)
        return score

    @staticmethod 
    def mcq_acc(answer, pred):
        periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
        commaStrip = re.compile("(\d)(\,)(\d)")
        punct = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]

        def processPunctuation(inText):
            outText = inText
            for p in punct:
                if (p + " " in inText or " " + p in inText) or (re.search(commaStrip, inText) != None):
                    outText = outText.replace(p, "")
                else:
                    outText = outText.replace(p, " ")
            outText = periodStrip.sub("", outText, re.UNICODE)
            return outText

        def process(answer):
            option_regex = re.compile(r"^([A-E])\.\s*(.+)$", re.IGNORECASE)
            match = option_regex.match(answer.strip())

            if match:
                # If matched, return the option letter in uppercase
                return match.group(1).upper()
            else:
                # If no match, process the answer as before
                answer = answer.replace("\n", " ")
                answer = answer.replace("\t", " ")
                answer = answer.strip()
                answer = processPunctuation(answer)
                answer = answer.strip("'")
                answer = answer.strip('"')
                answer = answer.strip(")")
                answer = answer.strip("(")
                answer = answer.strip().lower()

                # Try to find any single letter (A-E) in the processed answer
                letter_match = re.search(r"\b([A-E])\b", answer, re.IGNORECASE)
                if letter_match:
                    return letter_match.group(1).upper()

                return answer

        pred = process(pred)
        answer = process(answer)
        if pred not in ["A", "B", "C", "D"]:
            pred = "A"
        # print(pred, answer)
        if pred == answer:
            score = 1
        else:
            score = 0

        return score

def collate_fn(batches, tokenizer):
    pixel_values = torch.cat([_['pixel_values'] for _ in batches], dim=0)
    texts = [_['text'] for _ in batches]
    gts = [_['gt'] for _ in batches]
    num_patches_list = [_['num_patches'] for _ in batches]
    return pixel_values, texts, gts, num_patches_list


class RefCOCODataset(torch.utils.data.Dataset):

    def __init__(self, test, prompt, input_size=224, dynamic_image_size=False,
                 use_thumbnail=False, max_num=6, coco_dir=None):
        self.datas = json.load(open(test, "r"))
        self.prompt = prompt
        self.input_size = input_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num
        self.transform = build_transform(is_train=False, input_size=input_size)
        self.coco_dir = coco_dir

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]
        image = data['image']
        text = data['question']
        gt = data['answer']
        if os.path.exists(self.coco_dir):
            image  = os.path.join(self.coco_dir, os.path.basename(image))
        image = Image.open(image).convert('RGB')
        if self.dynamic_image_size:
            images = dynamic_preprocess(image, image_size=self.input_size,
                                        use_thumbnail=self.use_thumbnail,
                                        max_num=self.max_num)
        else:
            images = [image]
        pixel_values = [self.transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        return {
            'text': text,
            'gt': gt,
            'pixel_values': pixel_values,
            'num_patches': pixel_values.size(0)
        }


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def evaluate_chat_model():
    random.seed(args.seed)
    summaries = []

    for ds_name in args.datasets:
        dataset = RefCOCODataset(
            test=f"data/Mask_Perception/mask_perception.json",
            prompt="",
            input_size=image_size,
            dynamic_image_size=args.dynamic,
            use_thumbnail=use_thumbnail,
            max_num=args.max_num,
            coco_dir=args.coco_dir
        )
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=InferenceSampler(len(dataset)),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=partial(collate_fn, tokenizer=tokenizer),
        )

        outputs = []
        for _, (pixel_values, questions, gts, num_patches_list) in enumerate(tqdm(dataloader)):
            pixel_values = pixel_values.to(torch.bfloat16).cuda()
            generation_config = dict(
                num_beams=args.num_beams,
                max_new_tokens=100,
                min_new_tokens=1,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
            )
            pred = model.batch_chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                num_patches_list=num_patches_list,
                questions=questions,
                generation_config=generation_config
            )
            answers = pred

            for question, answer, gt in zip(questions, answers, gts):
                outputs.append({
                    'question': question,
                    'pred': answer.replace("<s> ", ""),
                    'gt': gt,
                })

        torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        merged_outputs = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_outputs, outputs)

        merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

        if torch.distributed.get_rank() == 0:
            print(f'Evaluating {ds_name} ...')
            time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
            results_file = f'rec_{ds_name}_{time_prefix}.json'
            results_file = os.path.join(args.out_dir, results_file)
            json.dump(merged_outputs, open(results_file, 'w'))

            evaluator = Scorer()
            for i, output in enumerate(merged_outputs):
                print(output)
                
                score = evaluator.get_a_score(
                    output['pred'],
                    output['gt'],
                    "choice_acc",
                )

            score_avg = evaluator.average()
            print(f'Accuracy: {score_avg}')
            summaries.append([args.checkpoint, ds_name, f'Accuracy: {score_avg} \n'])

        torch.distributed.barrier()

    out_path = '_'.join(args.checkpoint.split('/')[-2:])
    writer = open(os.path.join(args.out_dir, f'{out_path}.txt'), 'a')
    print(f"write results to file {os.path.join(args.out_dir, f'{out_path}.txt')}")
    for summary in summaries:
        print(summary)
        writer.write(f'{summary}\n')
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--datasets', type=str, default='mask_perception')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--num-beams', type=int, default=1)
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--sample', type=bool, default=False)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--coco-dir', type=str, default='./data/coco/train2014')
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--auto', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    args.datasets = args.datasets.split(',')
    print('datasets:', args.datasets)

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    PATTERN = re.compile(r'\[*\[(.*?),(.*?),(.*?),(.*?)\]\]*')
    model, tokenizer = load_model_and_tokenizer(args)
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    use_thumbnail = model.config.use_thumbnail

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    if total_params > 20 or args.dynamic:
        args.num_beams = 1
        print(f'[test] total_params: {total_params}B, use num_beams: {args.num_beams}')
    else:
        print(f'[test] total_params: {total_params}B')
    print(f'[test] image_size: {image_size}')
    print(f'[test] template: {model.config.template}')
    print(f'[test] dynamic_image_size: {args.dynamic}')
    print(f'[test] use_thumbnail: {use_thumbnail}')
    print(f'[test] max_num: {args.max_num}')

    evaluate_chat_model()
