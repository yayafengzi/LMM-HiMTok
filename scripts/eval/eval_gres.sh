export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python eval/evaluate_referseg.py \
    --datasets 'grefcoco_val,grefcoco_testA,grefcoco_testB' \
    --max-num 4 \
    --checkpoint yayafengzi/InternVL2_5-HiMTok-8B