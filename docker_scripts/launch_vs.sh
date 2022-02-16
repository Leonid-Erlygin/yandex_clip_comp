docker run\
     -d\
      --ipc=host\
      --rm\
       -v "$(realpath ..)":/home/devel/mlcup_cv\
        --runtime nvidia\
         --name ml_cv_vs\
          -p 4989:4989\
           -t ml_cv\
            /root/.local/bin/code-server\
             --bind-addr 0.0.0.0:4989\
              --auth none\
