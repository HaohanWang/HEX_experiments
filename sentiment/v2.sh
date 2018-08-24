# no hex
python cnn_run_v2.py -c status/ -e 100 -re 0 -corr 3 -hex 1 -save status/ -row 0 -col 1 -ng 16 -div 1000 > vanilla_c3.txt

#with hex
python cnn_run_v2.py -c status/ -e 100 -re 0 -corr 3 -hex 1 -save status/ -row 0 -col 1 -ng 16 -div 5 > hex_5_c3.txt

python cnn_run_v2.py -c status/ -e 100 -re 0 -corr 3 -hex 1 -save status/ -row 0 -col 1 -ng 16 -div 10 > hex_10_c3.txt

python cnn_run_v2.py -c status/ -e 100 -re 0 -corr 3 -hex 1 -save status/ -row 0 -col 1 -ng 16 -div 20 > hex_20_c3.txt

python cnn_run_v2.py -c status/ -e 100 -re 0 -corr 3 -hex 1 -save status/ -row 0 -col 1 -ng 16 -div 50 > hex_50_c3.txt

