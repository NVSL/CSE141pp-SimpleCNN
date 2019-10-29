USER_CFLAGS=-I..
include ../Make.rules

EXAMPLES=alexnet.exe toy.exe simple.exe
default: $(EXAMPLES)

%.exe : %.o 
	$(CXX) $(GENERIC_LDFLAGS) $^ -o $@

clean: examples-clean
examples-clean:
	rm -rf $(EXAMPLES)



