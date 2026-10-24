#!/bin/bash
# V56 on one AutoDL GPU: the 256-wide shared model and a 128-wide control (V54's size, same machine and
# precision as the 256 run), side by side. Run in the stage folder.
python -m venv /root/t && /root/t/bin/pip install -q numpy torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121 || exit 1
mkdir -p runs; export V56_DATA=data
/root/t/bin/python fit.py s256 --d 256 --layers 4 --heads 8 > runs/s256.out 2>&1 & a=$!
/root/t/bin/python fit.py s128 --d 128 --layers 4 --heads 4 > runs/s128.out 2>&1 & b=$!
wait $a; ra=$?; wait $b; rb=$?
[ $ra = 0 ] && [ $rb = 0 ]
