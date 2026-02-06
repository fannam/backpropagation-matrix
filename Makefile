CXX := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Iinclude
LDFLAGS := 

SRC := \
	src/main.cpp \
	src/core/Tensor.cpp \
	src/operators/BinaryOps.cpp \
	src/operators/UnaryOps.cpp \
	src/operators/MatrixOps.cpp

OBJ := $(SRC:.cpp=.o)

BIN := test_app

.PHONY: all run clean

all: $(BIN)

$(BIN): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJ) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(BIN)
	./$(BIN)

clean:
	rm -f $(OBJ) $(BIN)
