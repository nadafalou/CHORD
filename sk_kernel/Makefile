CC=/usr/local/cuda/bin/nvcc

DEPS = sk_kernel.cuh

OBJ = sk_kernel.o

%.o: %.cu $(DEPS)
	$(CC) -c -o $@ $< 

main: $(OBJ) main.o 
	$(CC) -o $@ $^ 

test: $(OBJ) test_downsample.o
	$(CC) -o $@ $^ 

.PHONY: clean

clean:
	rm -f *.o main test