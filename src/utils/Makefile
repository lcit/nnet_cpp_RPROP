# =========================================================================
# Author: Leonardo Citraro
# Company: 
# Filename: Makefile	
# =========================================================================
MAIN_DIR = ../../
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
OBJS        =  
LDFLAGS     = -g -pthread $(DEPS)
# /////////////////////////////////////////////////////////////////////////
all: utils.o

utils.o: utils.cpp
	$(CPP) $(CPPFLAGS) $(LDFLAGS) -c utils.cpp

clean:
	-rm -f *.o; find . -executable -type f -delete
