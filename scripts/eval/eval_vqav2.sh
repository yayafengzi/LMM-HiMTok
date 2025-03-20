export PYTHONPATH="${PYTHONPATH}:$(pwd)"

torchrun --nproc_per_node=1 --master_port=28585 eval/evaluate_vqa.py \
            --datasets 'vqav2_val'  \
            --dynamic --max-num 4 \
            --checkpoint yayafengzi/InternVL2_5-HiMTok-8B