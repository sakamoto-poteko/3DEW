CC=mpicc
CXX=mpicxx
LINKER=mpicxx

TARGETDIR=build/

COMMON= -O3 -openmp
CFLAGS= $(COMMON) -std=c99 $(PHI) -vec_report
CXXFLAGS= $(COMMON) -std=c++11 $(PHI) -vec_report
LDFLAGS= $(COMMON)

CFILES=$(wildcard *.c)
CPPFILES=$(wildcard *.cpp)

COBJS=$(CFILES:%.c=%.c-objs)
CPPOBJS=$(CPPFILES:%.cpp=%.cpp-objs)

TARGET=3dew

all: targetdir $(TARGET)

targetdir:
	mkdir -p $(TARGETDIR)

$(TARGET): $(COBJS) $(CPPOBJS)
	$(LINKER) $(LDFLAGS) $(addprefix $(TARGETDIR),$(COBJS)) $(addprefix $(TARGETDIR),$(CPPOBJS)) -o $(TARGETDIR)$(TARGET)

%.c-objs: %.c
	$(CC) -o $(TARGETDIR)$@ -c $(CFLAGS) $<

%.cpp-objs: %.cpp
	$(CXX) -o $(TARGETDIR)$@ -c $(CXXFLAGS) $<

clean:
	rm -rf $(addprefix $(TARGETDIR),$(COBJS)) $(addprefix $(TARGETDIR),$(CPPOBJS))
