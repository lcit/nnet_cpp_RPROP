# =========================================================================
# Author: Leonardo Citraro
# Company: 
# Filename: Makefile	
# =========================================================================
MAIN_DIR = ../
# /////////////////////////////////////////////////////////////////////////
SRC_DIR = $(MAIN_DIR)/src
INC_DIR = $(MAIN_DIR)/include
LIB_DIR = $(MAIN_DIR)/lib
BIN_DIR = $(MAIN_DIR)/bin
# /////////////////////////////////////////////////////////////////////////

# /////////////////////////////////////////////////////////////////////////
UTILS_DIR	= utils
# /////////////////////////////////////////////////////////////////////////
CPP         = g++
INCLUDES    = -I. -I$(INC_DIR)/$(UTILS_DIR)
CPPFLAGS    = -std=c++14 -pthread $(INCLUDES)
DESP        = 
OBJS        = $(SRC_DIR)/utils/Eigen_plus.o $(SRC_DIR)/utils/utils.o
LDFLAGS     = -g -pthread $(DEPS)
# /////////////////////////////////////////////////////////////////////////
all: nnet

nnet: nnet.cpp
	$(CPP) $(CPPFLAGS) -o nnet nnet.cpp $(OBJS) $(LDFLAGS)

clean:
	-rm -f *.o; rm nnet
