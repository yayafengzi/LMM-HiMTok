export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python eval/evaluate_referseg.py \
    --datasets 'refcoco_val,refcoco_testA,refcoco_testB,refcoco+_val,refcoco+_testA,refcoco+_testB,refcocog_val,refcocog_test' \
    --max-num 4 \
    --checkpoint yayafengzi/InternVL2_5-HiMTok-8B \
    --checkpoint-sam yayafengzi/InternVL2_5-HiMTok-8B/sam.pth