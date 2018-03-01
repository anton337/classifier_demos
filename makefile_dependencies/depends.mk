GCC=g++
IGNORE=-Wno-write-strings
CPPFLAGS=-O3 -g0 $(IGNORE)
#CPPFLAGS=-O0 -g3 $(IGNORE)
INCLUDE_DIRECTORY=$(PROJECT_HOME)/include/
INCLUDES=	-I$(INCLUDE_DIRECTORY)visualization\
			-I$(INCLUDE_DIRECTORY)file_input\
			-I$(INCLUDE_DIRECTORY)machine_learning_tools\
			-I$(INCLUDE_DIRECTORY)snapshots\
			-I$(INCLUDE_DIRECTORY)diagnostics\
			-I$(INCLUDE_DIRECTORY)text_tools\
			-I$(INCLUDE_DIRECTORY)merge_tools
SOURCES=    diagnostics/Histogram.cpp\
			file_input/readBMP.cpp
LIBS=       -lglut -lGL -lGLU -lboost_system -lboost_thread
