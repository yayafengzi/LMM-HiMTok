export PYTHONPATH="${PYTHONPATH}:$(pwd)"

torchrun --nproc_per_node=1 --master_port=29594 eval/pope/evaluate_pope.py \
            --checkpoint yayafengzi/InternVL2_5-HiMTok-8B --dynamic --max-num 4
