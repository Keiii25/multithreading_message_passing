ALL: main

main: main.c 
	mpicc -Wall main.c seafloor_sensor.c base_station.c -o mainOut -lm -fopenmp

run:
	mpirun -oversubscribe -np 10 mainOut 3 3 5 100
	
clean :
	/bin/rm -f mainOut *.o
	/bin/rm -f result_log *.txt
	/bin/rm -f sentinel *.txt