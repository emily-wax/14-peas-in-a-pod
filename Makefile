hello: cuda.o mpi.o
	mpicc -o hello cuda.o mpi.o -lcudart

cuda.o: cuda.cu
	nvcc -c cuda.cu
	
mpi.o: mpi.c
	mpicc -c mpi.c
	
clean:
	rm *.o hello