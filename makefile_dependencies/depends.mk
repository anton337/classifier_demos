GCC=g++
CPPFLAGS=-O3 -g0
INCLUDE_DIRECTORY=$(PROJECT_HOME)/include/
INCLUDES=	-I$(INCLUDE_DIRECTORY)visualization\
			-I$(INCLUDE_DIRECTORY)file_input
SOURCES=    diagnostics/Histogram.cpp\
			file_input/readBMP.cpp
LIBS=       -lglut -lGL -lGLU -lboost_system -lboost_thread
