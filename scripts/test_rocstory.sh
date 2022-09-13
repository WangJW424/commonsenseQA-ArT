export DATA_DIR=../dataset/rocstory
export GPT_DIR=../gpts
python ../src/evaluating_with_notes.py \
  --model_type gpt2 \
  --model_name_or_path $GPT_DIR/gpt2-medium \
  --task_name rocstory \
  --data_dir $DATA_DIR \
  --output_dir $DATA_DIR/outputs \
  --eval_file dev_with_5_keyphrases_32_notes_gpt2-medium.jsonl \
  --max_seq_length 256 \
  --attaching_side right \
  --take_notes \
  --scale_type mean \
  --max_notes_num 32 \
  --batch_size_per_gpu 8 \
  --overwrite_output_file \
  --device_ids 0
