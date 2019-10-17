include ../Make.rules

EXAMPLES=example1.exe example2.exe alexnet.exe toy.exe
default: $(EXAMPLES)

%.exe : %.o 
	$(CXX) $(USER_LDFLAGS) $(LD_OPTS) $^ -o $@

clean: examples-clean
examples-clean:
	rm -rf $(EXAMPLES)



