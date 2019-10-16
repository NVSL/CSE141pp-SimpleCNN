include ../Make.rules

TOOLS=build_mnist.exe build_imagenet.exe
default: $(TOOLS)

%.exe : %.o 
	$(CXX) $(USER_LDFLAGS) $(LD_OPTS) $^ -ljpeg -o $@

examples-clean:
	rm -rf $(TOOLS)

clean: examples-clean
