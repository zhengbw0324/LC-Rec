
python -u main.py \
  --lr 1e-3 \
  --epochs 10000 \
  --batch_size 1024 \
  --weight_decay 1e-4 \
  --lr_scheduler_type linear \
  --dropout_prob 0.0 \
  --bn False \
  --e_dim 32 \
  --quant_loss_weight 1.0 \
  --beta 0.25 \
  --num_emb_list 256 256 256 256 \
  --sk_epsilons 0.0 0.0 0.0 0.003 \
  --layers 2048 1024 512 256 128 64 \
  --device cuda:0 \
  --data_path /data/Games/Games.emb-llama-td.npy \
  --ckpt_dir ./ckpt/