docker run\
 --rm\
 --ipc=host\
 --name index_download\
 -v "$(realpath ..)":/home/devel/mlcup_cv\
 --workdir /home/devel/mlcup_cv/download_dataset\
 -t ml_cv\
 python download_yandexdataset.py