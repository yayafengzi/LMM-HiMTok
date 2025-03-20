# README for Evaluation

## Data Preparation

Before starting to download the data, please create the `data` folder.

### COCO Images

Follow the instructions below to prepare the data:

```shell
# Step 1: Create the data directory
mkdir -p data/coco && cd data/coco

# Step 2: Download and unzip image files
wget http://images.cocodataset.org/zips/train2014.zip && unzip train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip && unzip val2014.zip
```

After preparation is complete, the directory structure is:

```shell
data/coco
├── train2014
├── val2014
```

### (Generalized) Referring Expression Segmentation

Follow the instructions below to prepare the data:
```shell
# Step 1: Create the data directory
mkdir -p data/res && cd data/res
```

Download converted files (by PSALM) ([Google Drive](https://drive.google.com/file/d/1EcC1tl1OQRgIqqy7KFG7JZz2KHujAQB3/view?usp=sharing) | [Baidu Cloud](https://pan.baidu.com/s/1NRGJGkJDUGn8CU-sU5ScOg) (code: hust)).

After preparation is complete, the directory structure is:
```shell
refcoco/
    refcoco_val.json
    refcoco_testA.json
    refcoco_testB.json
refcoco+/
    refcoco+_val.json
    refcoco+_testA.json
    refcoco+_testB.json
refcocog/
    refcocog_val.json
    refcocog_test.json
grefcoco/
    refcocog_val.json
    refcocog_testA.json
    refcocog_testB.json
```

### Reasoning Segmentation

Follow the instructions below to prepare the data:

Download and extract the data to the `data` folder: [ReasonSeg](https://drive.google.com/drive/folders/125mewyg5Ao6tZ3ZdJ-1-E3n04LGVELqy?usp=sharing)

After preparation is complete, the directory structure is:

```shell
ReasonSeg/
    train/
        image_1.jpg, image_1.json
        image_2.jpg, image_2.json
    val/
        image_1.jpg, image_1.json
        image_2.jpg, image_2.json
```

### Referring Expression Comprehension

Follow the instructions below to prepare the data:

```shell
# Step 1: Create the data directory
mkdir -p data/rec && cd data/rec

# Step 2: Download converted files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/refcoco/refcoco_val.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/refcoco/refcoco_testA.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/refcoco/refcoco_testB.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/refcoco%2B/refcoco%2B_val.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/refcoco%2B/refcoco%2B_testA.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/refcoco%2B/refcoco%2B_testB.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/refcocog/refcocog_val.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/refcocog/refcocog_test.jsonl

cd ../..
```

After preparation is complete, the directory structure is:

```shell
data/rec
├── refcocog_test.jsonl
├── refcocog_val.jsonl
├── refcoco_testA.jsonl
├── refcoco+_testA.jsonl
├── refcoco_testB.jsonl
├── refcoco+_testB.jsonl
├── refcoco_val.jsonl
└── refcoco+_val.jsonl
```

### MME

Follow the instructions below to prepare the data:

```shell
# Step 1: Create the data directory
mkdir -p data/mme && cd data/mme

# Step 2: Download MME_Benchmark_release_version
wget https://huggingface.co/OpenGVLab/InternVL/resolve/main/MME_Benchmark_release_version.zip
unzip MME_Benchmark_release_version.zip

cd ../..
```

After preparation is complete, the directory structure is:

```shell
data/mme
 └── MME_Benchmark_release_version
```

### VQAv2

Follow the instructions below to prepare the data:

```shell
# Step 1: Create the data directory
mkdir -p data/vqav2 && cd data/vqav2

# Step 2: Make sure you have downloaded COCO images
ln -s ../coco/train2014 ./
ln -s ../coco/val2014 ./
ln -s ../coco/test2015 ./

# Step 3: Download questions and annotations
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip && unzip v2_Annotations_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip && unzip v2_Questions_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip && unzip v2_Annotations_Val_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip && unzip v2_Questions_Val_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip && unzip v2_Questions_Test_mscoco.zip

# Step 4: Download converted files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vqav2/vqav2_train.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vqav2/vqav2_val.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vqav2/vqav2_testdev.jsonl

cd ../..
```

After preparation is complete, the directory structure is:

```shell
data/vqav2
├── train2014 -> ../coco/train2014
├── val2014 -> ../coco/val2014
├── test2015 -> ../coco/test2015
├── v2_mscoco_train2014_annotations.json
├── v2_mscoco_train2014_complementary_pairs.json
├── v2_mscoco_val2014_annotations.json
├── v2_OpenEnded_mscoco_test2015_questions.json
├── v2_OpenEnded_mscoco_test-dev2015_questions.json
├── v2_OpenEnded_mscoco_train2014_questions.json
├── v2_OpenEnded_mscoco_val2014_questions.json
├── vqav2_testdev.jsonl
├── vqav2_train.jsonl
└── vqav2_val.jsonl
```

### POPE

Follow the instructions below to prepare the data:

```shell
# Step 1: Create the data directory
mkdir -p data/pope && cd data/pope

# Step 2: Make sure you have downloaded COCO images
ln -s ../coco/val2014 ./
wget https://github.com/OpenGVLab/InternVL/releases/download/data/llava_pope_test.jsonl

# Step 3: Download `coco` from POPE
mkdir -p coco && cd coco
wget https://github.com/AoiDragon/POPE/raw/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco/coco_pope_adversarial.json
wget https://github.com/AoiDragon/POPE/raw/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco/coco_pope_popular.json
wget https://github.com/AoiDragon/POPE/raw/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco/coco_pope_random.json
cd ../../..
```

After preparation is complete, the directory structure is:

```shell
data/pope
├── coco
│   ├── coco_pope_adversarial.json
│   ├── coco_pope_popular.json
│   └── coco_pope_random.json
├── llava_pope_test.jsonl
└── val2014
```

### Mask Perception

Download the data from [here](https://huggingface.co/datasets/yayafengzi/Mask_Perception) and place it in the `data` folder.

## Evaluation

Run the following scripts:

```shell
bash scripts/eval_res.sh
bash scripts/eval_res_with_sam.sh
bash scripts/eval_gres.sh
bash scripts/eval_reasonseg.sh
bash scripts/eval_rec.sh
bash scripts/eval_mme.sh
bash scripts/eval_vqav2.sh
bash scripts/eval_pope.sh
bash scripts/eval_map.sh
```
