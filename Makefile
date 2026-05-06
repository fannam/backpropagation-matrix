CXX := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Iinclude -Itests
LDFLAGS := 

LIB_SRC := \
	src/core/Tensor.cpp \
	src/activations/Activations.cpp \
	src/operators/BinaryOps.cpp \
	src/operators/UnaryOps.cpp \
	src/operators/MatrixOps.cpp \
	src/operators/ReductionOps.cpp \
	src/loss/Loss.cpp

TEST_SRC := \
	tests/main.cpp \
	tests/test_tensor.cpp \
	tests/test_ops.cpp \
	tests/test_activations.cpp

SRC := $(LIB_SRC) $(TEST_SRC)
OBJ := $(SRC:.cpp=.o)

BIN := test_app
DEMO_BIN := demo_app
DEMO_OBJ := $(LIB_SRC:.cpp=.o) src/main.o

.PHONY: all run demo clean

all: $(BIN)

$(BIN): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJ) $(LDFLAGS)

$(DEMO_BIN): $(DEMO_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $(DEMO_OBJ) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(BIN)
	./$(BIN)

demo: $(DEMO_BIN)
	./$(DEMO_BIN)

clean:
	rm -f $(OBJ) src/main.o $(BIN) $(DEMO_BIN)
