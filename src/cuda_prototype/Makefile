CXX = nvcc
CUFILES = main.cu
HEADERS = helpers.cuh matmul.cuh matmul-tensor-naive.cuh matmul-tensor.cuh matmul-cutlass.cuh
OPTIONS = -O3 -std=c++17 -arch=sm_80 -Xptxas=-v --expt-relaxed-constexpr -lcublas -I../../cutlass/include/ #-Wall  -pedantic

MAIN = main

# depends on building the main file
all: $(MAIN)

run: $(MAIN)
	./$(MAIN)

# depends on headers and source
$(MAIN): $(CUFILES) $(HEADERS)
	$(CXX) $(OPTIONS) $(CUFILES) -o $(MAIN)

clean:
	- rm main
