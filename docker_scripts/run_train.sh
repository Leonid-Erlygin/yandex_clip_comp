docker run\
 --rm\
 --ipc=host\
 --runtime nvidia\
 --name train_cv\
 -v "$(realpath ..)":/home/devel/mlcup_cv\
 --workdir /home/devel/mlcup_cv/baseline\
 -t ml_cv\
 python meta_train.py
