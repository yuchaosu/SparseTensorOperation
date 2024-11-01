# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++11 -fopenmp -Wall -O2 -Iinclude

# Target executable
TARGET = bin/tensor_program

# Source files
SRCS = src/main.cpp \
       src/SparseTensorCOO.cpp \
       src/HiCOOTensor.cpp \
       src/TensorOperations.cpp

# Object files
OBJS = $(SRCS:src/%.cpp=obj/%.o)

# Create necessary directories
$(shell mkdir -p bin)
$(shell mkdir -p obj)

# Default rule
all: $(TARGET)

# Linking
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

# Compilation
obj/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up
clean:
	rm -f obj/*.o $(TARGET)

# Phony targets
.PHONY: all clean
