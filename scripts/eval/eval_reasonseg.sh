export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python eval/evaluate_reasonseg.py \
    --datasets 'reasonseg_val,reasonseg_test' \
    --max-num 4 \
    --checkpoint yayafengzi/InternVL2_5-HiMTok-8B