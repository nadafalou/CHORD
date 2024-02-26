CUDA_PATH = /usr/local/cuda
NVCC = $(CUDA_PATH)/bin/nvcc
NCU = $(CUDA_PATH)/bin/ncu
ARCH = -arch=sm_86
CFLAGS = -g -O2
# LDFLAGS =

DEPS = sk_kernel.cuh mask.cuh
OBJ = sk_kernel.o mask.o

%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(CFLAGS) -c -o $@ $<

main: $(OBJ) main.o
	$(NVCC) $(ARCH) $(CFLAGS) $(LDFLAGS) -o $@ $^

test: $(OBJ) test.o
	$(NVCC) $(ARCH) $(CFLAGS) $(LDFLAGS) -o $@ $^

profile: main
	$(NCU) --set full --metrics all -f -o ./output.ncu-rep $^ --kernel-name $(KERNEL)

.PHONY: clean

clean:
	rm -f *.o main test