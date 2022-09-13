export DATA_DIR=../dataset/rocstory
export GPT_DIR=../gpts

python ../SIFRanker/extract_keyphrases.py \
  --task_name rocstory \
  --data_dir $DATA_DIR \
  --input_file dev.jsonl \
  --options_file ../SIFRanker/elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json \
  --weight_file ../SIFRanker/elmo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5 \
  --word_weight_file ../SIFRanker/auxiliary_data/enwiki_vocab_min200.txt \
  --corenlp_file ../SIFRanker/corenlp/stanford-corenlp-4.2.2 \
  --max_keyphrase_num 5

python ../src/note_generating.py \
  --model_type gpt2 \
  --model_name_or_path $GPT_DIR/gpt2-medium \
  --task_name rocstory \
  --data_dir $DATA_DIR \
  --output_dir $DATA_DIR/outputs \
  --input_file dev_with_5_keyphrases.jsonl \
  --top_p=0.8 \
  --max_seq_length 256 \
  --num_once_gen 32 \
  --max_gen_length 15 \
  --max_gen_time 3 \
  --return_notes_num 32 \
  --batch_size_per_gpu 8 \
  --do_rethink \
  --overwrite_output_file \
  --device_ids 0
