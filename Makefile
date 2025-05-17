LLVM_PATH = $(shell brew --prefix llvm)
GTEST_PATH = $(shell brew --prefix googletest)

CXX = $(LLVM_PATH)/bin/clang++
CXXFLAGS = -std=c++23 -fopenmp -O3 -Iutils -I$(GTEST_PATH)/include -Wall -Wextra -pedantic
LDFLAGS = -L$(LLVM_PATH)/lib -lomp -L$(GTEST_PATH)/lib -lgtest -lgtest_main -pthread

UTILS_DIR = utils
TEST_DIR = tests
BIN_DIR = bin
OBJ_DIR = obj

SRCS = $(wildcard $(UTILS_DIR)/*.cpp)
OBJS = $(patsubst $(UTILS_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRCS))
TEST_SRCS = $(wildcard $(TEST_DIR)/*.cpp)
TEST_OBJS = $(patsubst $(TEST_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(TEST_SRCS))
TARGET = $(BIN_DIR)/linalg_test

all: $(TARGET)

$(TARGET): $(OBJS) $(TEST_OBJS)
	mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(UTILS_DIR)/%.cpp $(UTILS_DIR)/%.h
	mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(TEST_DIR)/%.cpp
	mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

.PHONY: all clean