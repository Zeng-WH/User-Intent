export TASK_NAME=mrpc
export HF_DATASETS_CACHE="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/zengweihao02/cache"
CUDA_VISIBLE_DEVICES=1 python3 run_intent_gpt_search_fgm.py \
  --model_name_or_path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/zengweihao02/SereTOD/SereTOD/SereTOD2022-main/Track2/intent_class/checkpoint/gpt2-serach_datav1-fgm \
  --do_eval \
  --train_file /mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/zengweihao02/SereTOD/SereTOD2022-main/Track2/intent_class/data/datav1/train.json \
  --validation_file /mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/zengweihao02/SereTOD/SereTOD2022-main/Track2/intent_class/data/datav1/test.json \
  --max_seq_length 128 \
  --eval_steps=50 \
  --logging_steps=10 \
  --per_device_train_batch_size 64 \
  --learning_rate 2e-5 \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --load_best_model_at_end True \
  --num_train_epochs 5 \
  --overwrite_output_dir \
  --report_to="tensorboard" \
  --prob_gate=0.5 \
  --output_dir /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/zengweihao02/SereTOD/SereTOD/SereTOD2022-main/Track2/intent_class/checkpoint/gpt2-serach_datav1-fgm \