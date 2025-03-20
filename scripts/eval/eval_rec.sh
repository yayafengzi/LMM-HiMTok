export PYTHONPATH="${PYTHONPATH}:$(pwd)"

torchrun --nproc_per_node=1 --master_port=28584 eval/evaluate_grounding.py \
            --datasets 'refcoco_val,refcoco_testA,refcoco_testB,refcoco+_val,refcoco+_testA,refcoco+_testB,refcocog_val,refcocog_test' \
            --dynamic --max-num 4 \
            --checkpoint yayafengzi/InternVL2_5-HiMTok-8B