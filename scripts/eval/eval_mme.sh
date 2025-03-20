export PYTHONPATH="${PYTHONPATH}:$(pwd)"

CHECKPOINT=yayafengzi/InternVL2_5-HiMTok-8B
DIRNAME=`basename ${CHECKPOINT}`
cd eval/mme
python eval.py --checkpoint ${CHECKPOINT} --dynamic --max-num 4
python calculation.py --results_dir ${DIRNAME}
cd ../../
