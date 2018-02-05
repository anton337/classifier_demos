include $(PROJECT_HOME)/makefile_dependencies/depends.mk

#OBJECT_DIRECTORY=$(PROJECT_HOME)/obj/

#SOURCE_DIRECTORY=$(PROJECT_HOME)/src/
#
#TARGET_DIRECTORY=$(PROJECT_HOME)/bin/
#
#BUILD_TARGET=$(PROJECT)
#
#EXE_NAME=/demo.bin
#
#TARGET=$(TARGET_DIRECTORY)$(BUILD_TARGET)$(EXE_NAME)
#
#PROJECT_SOURCE=	$(PROJECT_HOME)/src/projects/$(BUILD_TARGET)/main.cpp
#
#OBJECTS=$(join $(addprefix $(OBJECT_DIRECTORY), $(dir $(SOURCES))), $(notdir $(SOURCES:.cpp=.o)))
#
#OBJECT_PATH=$(PROJECT_HOME)/obj/
#
#all: $(OBJECTS) $(TARGET)
#
#$(TARGET): $(OBJECTS)
#	echo "Objects:""$(OBJECTS)"
#	echo "Sources:""$(SOURCES)"
#	$(GCC) $(CPPFLAGS) $(INCLUDES) $(OBJECTS) $(PROJECT_SOURCE) -o $@ $(LIBS)
#
#$(OBJECTS): $(OBJECT_PATH)%.o : $(SOURCE_PATH)%.cpp
#	$(GCC) $(CPPFLAGS) $(INCLUDES) -c $< -o $@
#
#clean:
#	find $(PROJECT_HOME) -name "*.o" | xargs rm -f
#	find $(OBJECT_DIRECTORY) -name "*.o" | xargs rm -f
#	rm -f $(TARGET)
#
#

#g++ -O3 -g0 -I/home/antonk/classifier_demos/classifier_demos/include/visualization -I/home/antonk/classifier_demos/classifier_demos/include/file_input -c diagnostics/Histogram.cpp -o /home/antonk/classifier_demos/classifier_demos/obj/diagnostics/Histogram.o
#g++ -O3 -g0 -I/home/antonk/classifier_demos/classifier_demos/include/visualization -I/home/antonk/classifier_demos/classifier_demos/include/file_input -c diagnostics/Histogram.cpp -o /home/antonk/classifier_demos/classifier_demos/obj/file_input/readBMP.o
#echo "Objects:""/home/antonk/classifier_demos/classifier_demos/obj/diagnostics/Histogram.o /home/antonk/classifier_demos/classifier_demos/obj/file_input/readBMP.o"
#	Objects:/home/antonk/classifier_demos/classifier_demos/obj/diagnostics/Histogram.o /home/antonk/classifier_demos/classifier_demos/obj/file_input/readBMP.o
#echo "Sources:""diagnostics/Histogram.cpp file_input/readBMP.cpp"
#	Sources:diagnostics/Histogram.cpp file_input/readBMP.cpp
#	g++ -O3 -g0 -I/home/antonk/classifier_demos/classifier_demos/include/visualization -I/home/antonk/classifier_demos/classifier_demos/include/file_input /home/antonk/classifier_demos/classifier_demos/obj/diagnostics/Histogram.o /home/antonk/classifier_demos/classifier_demos/obj/file_input/readBMP.o /home/antonk/classifier_demos/classifier_demos/src/projects/simple_ann_classification/main.cpp -o /home/antonk/classifier_demos/classifier_demos/bin/simple_ann_classification/demo.bin -lglut -lGL -lGLU -lboost_system -lboost_thread
#

OBJECT_DIRECTORY=$(PROJECT_HOME)/obj/

SOURCE_DIRECTORY=$(PROJECT_HOME)/src/

TARGET_DIRECTORY=$(PROJECT_HOME)/bin/

BUILD_TARGET=$(PROJECT)

EXE_NAME=/demo.bin

TARGET=$(TARGET_DIRECTORY)$(BUILD_TARGET)$(EXE_NAME)

PROJECT_SOURCE=	$(PROJECT_HOME)/src/projects/$(BUILD_TARGET)/main.cpp

OBJECTS=$(OBJECT_DIRECTORY)/diagnostics/Histogram.o $(OBJECT_DIRECTORY)/file_input/readBMP.o

all: $(OBJECTS) $(TARGET)

$(OBJECT_DIRECTORY)/diagnostics/Histogram.o: 
	$(GCC) $(CPPFLAGS) $(INCLUDES) -c $(SOURCE_DIRECTORY)/diagnostics/Histogram.cpp -o $(OBJECT_DIRECTORY)/diagnostics/Histogram.o

$(OBJECT_DIRECTORY)/file_input/readBMP.o: 
	$(GCC) $(CPPFLAGS) $(INCLUDES) -c $(SOURCE_DIRECTORY)/file_input/readBMP.cpp -o $(OBJECT_DIRECTORY)/file_input/readBMP.o

$(TARGET): $(OBJECTS)
	$(GCC) $(CPPFLAGS) $(INCLUDES) $(PROJECT_SOURCE) $(OBJECTS) -o $(TARGET) $(LIBS)

clean:
	find $(OBJECT_DIRECTORY) -name "*.o" | xargs rm -f
	rm -f $(TARGET)


