CC=/usr/local/cuda/bin/nvcc

DEPS = sk_kernel.cuh mask.cuh

OBJ = sk_kernel.o mask.o

%.o: %.cu $(DEPS)
	$(CC) -g -G -c -o $@ $< 

main: $(OBJ) main.o 
	$(CC) -o $@ $^ 

test: $(OBJ) test.o
	$(CC) -g -G -o $@ $^ 

.PHONY: clean

clean:
	rm -f *.o main test