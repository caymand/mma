CXX = nvcc
CUFILES = main.cu
HEADERS = helpers.cuh matmul.cuh matmul-tensor-naive.cuh matmul-tensor.cuh
OPTIONS = -O3 -std=c++17 -arch=sm_80 -Xptxas=-v --expt-relaxed-constexpr -lcublas #-Wall  -pedantic

MAIN = main
GOLDEN_CPU = golden_cpu.o

# depends on building the main file
all: $(MAIN)

run: $(MAIN)
	./$(MAIN)

# depends on headers and source
$(MAIN): $(CFILES) $(HEADERS)
	$(CXX) $(OPTIONS) $(CUFILES) -o $(MAIN)

clean:
	- rm main
