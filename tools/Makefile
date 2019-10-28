USER_CFLAGS=-I..
include ../Make.rules

TOOLS=build_mnist.exe build_imagenet.exe
default: $(TOOLS)

%.exe : %.o 
	$(CXX) $(USER_LDFLAGS) $(LD_OPTS) $^ -ljpeg -o $@

.PHONY:tools-clean
tools-clean:
	rm -rf $(TOOLS)
clean: tools-clean
