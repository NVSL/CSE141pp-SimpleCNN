include ../Make.rules

TOOLS=build_mnist.exe
default: $(TOOLS)

%.exe : %.o 
	$(CXX) $(USER_LDFLAGS) $(LD_OPTS) $^ -o $@

examples-clean:
	rm -rf $(TOOLS)

clean: examples-clean
