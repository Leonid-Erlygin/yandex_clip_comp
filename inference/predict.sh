PYTHONPATH=./py_packages:$PYTHONPATH PYTHONDONTWRITEBYTECODE=1 python3 predict_onnx_with_rubert.py \
  --model_name post_7h \
  --data_directory $1 --predicts_file $2
