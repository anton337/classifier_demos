include $(PROJECT_HOME)/makefile_dependencies/depends.mk

OBJECT_DIRECTORY=$(PROJECT_HOME)/obj/

TARGET_DIRECTORY=$(PROJECT_HOME)/bin/

BUILD_TARGET=$(PROJECT)

EXE_NAME=/demo.bin

TARGET=$(TARGET_DIRECTORY)$(BUILD_TARGET)$(EXE_NAME)

PROJECT_SOURCE=	projects/$(BUILD_TARGET)/main.cpp

OBJECTS=$(join $(OBJECT_DIRECTORY), $(notdir $(SOURCES:.cpp=.o)))

all: $(OBJECTS) $(TARGET)

$(TARGET): $(OBJECTS)
	echo "Objects:""$(OBJECTS)"
	$(GCC) $(CPPFLAGS) $(INCLUDES) $(OBJECTS) $(PROJECT_SOURCE) -o $@ $(LIBS)

$(OBJECTS): $(SOURCES)
	$(GCC) $(CPPFLAGS) $(INCLUDES) -c $< -o $@

clean:
	find $(PROJECT_HOME) -name "*.o" | xargs rm -f
	find $(OBJECT_DIRECTORY) -name "*.o" | xargs rm -f
	rm -f $(TARGET)

