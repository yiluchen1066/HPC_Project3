export OMP_NUM_THREADS=1
./main 256 200 0.01

export OMP_NUM_THREADS=2
./main 256 200 0.01

export OMP_NUM_THREADS=4
./main 256 200 0.01

export OMP_NUM_THREADS=8
./main 256 200 0.01

export OMP_NUM_THREADS=16 
./main 256 200 0.01

