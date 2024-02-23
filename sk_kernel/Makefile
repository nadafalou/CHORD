CC=/usr/local/cuda/bin/nvcc -arch=sm_86

DEPS = sk_kernel.cuh mask.cuh small.cuh

OBJ = sk_kernel.o mask.o small.o

%.o: %.cu $(DEPS)
	$(CC) -g -c -O 2 -o $@ $< 

main: $(OBJ) main.o 
	$(CC) -O 2 -o $@ $^ 

test: $(OBJ) test.o
	$(CC) -g -o $@ $^ 

.PHONY: clean

clean:
	rm -f *.o main test