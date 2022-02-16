PYTHONPATH=./py_packages:$PYTHONPATH python3 predict_onnx_with_emb.py \
  --ckpt_path checkpoints/train_emb.ckpt \
  --data_directory $1 --predicts_file $2
