#!/bin/bash
# Same as gqa.sh but supports additional args and flexible ckpt like other *_flexible.sh scripts

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}
DEFAULT_CKPT="llava-v1.5-13b"

if [[ -z "$CKPT" ]]; then
    CKPT=$DEFAULT_CKPT
    echo "Using default CKPT=$CKPT"
fi

if [[ -z "$EXTRA_ARGS" ]]; then
    EXTRA_ARGS=""
fi

SPLIT="llava_gqa_testdev_balanced"
GQADIR="./playground/data/eval/gqa/data"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader_flexible \
        --model-path liuhaotian/$CKPT \
        --question-file ./playground/data/eval/gqa/$SPLIT.jsonl \
        --image-folder ./playground/data/eval/gqa/data/images \
        --answers-file ./playground/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        $EXTRA_ARGS &
done

wait

output_file=./playground/data/eval/gqa/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

cd $GQADIR
python eval/eval.py --tier testdev_balanced
