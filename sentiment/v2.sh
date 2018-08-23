# total hex
python cnn_run_v2.py -c status/ -e 1000 -re 0 -corr 8 -hex 1 -save status/ -row 0 -col 1 -ng 16 -div 0

#total no hex
python cnn_run_v2.py -c status/ -e 1000 -re 0 -corr 8 -hex 0 -save status/ -row 0 -col 1 -ng 16 -div 1000
