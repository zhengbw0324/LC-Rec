


DATASET=Games
DATA_PATH=./data
OUTPUT_DIR=./ckpt/$DATASET/
RESULTS_FILE=./results/$DATASET/xxx.json

torchrun --nproc_per_node=8 --master_port=23324 test_ddp.py \
    --ckpt_path $CKPT_PATH \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --results_file $RESULTS_FILE \
    --test_batch_size 1 \
    --num_beams 20 \
    --test_prompt_ids all \
    --index_file .index.json

