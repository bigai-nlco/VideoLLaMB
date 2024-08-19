

CKPT_NAME="llava-7b-ft-rmt1x-lvcn_32_8_pool12_new"
model_path="checkpoints/${CKPT_NAME}"
# CKPT_NAME="llava-vit-vivit-7b"
model_base="checkpoints/llava-v1.5-7b"
# model_base=None
cache_dir="./cache_dir"
GPT_Zero_Shot_QA="./playground/eval/GPT_Zero_Shot_QA"
video_dir="${GPT_Zero_Shot_QA}/EgoPlan_Zero_Shot_QA/videos"
gt_file_question="${GPT_Zero_Shot_QA}/EgoPlan_Zero_Shot_QA/test_q.json"
gt_file_answers="${GPT_Zero_Shot_QA}/EgoPlan_Zero_Shot_QA/test_a.json"
output_dir="${GPT_Zero_Shot_QA}/EgoPlan_Zero_Shot_QA/${CKPT_NAME}"
NUM_FRAMES=16

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

    #   --model_base ${model_base} \

for IDX in $(seq 0 $((CHUNKS-1))); do
  CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_videoqa_mc \
      --model_path ${model_path} \
      --cache_dir ${cache_dir} \
      --video_dir ${video_dir} \
      --gt_file_question ${gt_file_question} \
      --gt_file_answers ${gt_file_answers} \
      --output_dir ${output_dir} \
      --output_name ${CHUNKS}_${IDX} \
      --num_frames $NUM_FRAMES \
      --num_chunks $CHUNKS \
      --chunk_idx $IDX &
done

wait

output_file=${output_dir}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${output_dir}/${CHUNKS}_${IDX}.json >> "$output_file"
done

python scripts/eval_acc.py --src $output_file --dst $GQADIR/subset.json
