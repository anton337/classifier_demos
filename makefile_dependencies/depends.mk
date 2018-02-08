GCC=g++
CPPFLAGS=-O3 -g0
#CPPFLAGS=-O0 -g3
INCLUDE_DIRECTORY=$(PROJECT_HOME)/include/
INCLUDES=	-I$(INCLUDE_DIRECTORY)visualization\
			-I$(INCLUDE_DIRECTORY)file_input\
			-I$(INCLUDE_DIRECTORY)machine_learning_tools\
			-I$(INCLUDE_DIRECTORY)snapshots
SOURCES=    diagnostics/Histogram.cpp\
			file_input/readBMP.cpp
LIBS=       -lglut -lGL -lGLU -lboost_system -lboost_thread
