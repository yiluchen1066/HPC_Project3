export OMP_NUM_THREADS=1
./main 128 100 0.005

export OMP_NUM_THREADS=2
./main 128 100 0.005

export OMP_NUM_THREADS=4
./main 128 100 0.005

export OMP_NUM_THREADS=8
./main 128 100 0.005

export OMP_NUM_THREADS=16 
./main 128 100 0.005 

