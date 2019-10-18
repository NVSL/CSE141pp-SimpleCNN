include ../Make.rules

EXAMPLES=example1.exe example2.exe alexnet.exe toy.exe perceptron.exe
default: $(EXAMPLES)

%.exe : %.o 
	$(CXX) $(GENERIC_LDFLAGS) $^ -o $@

clean: examples-clean
examples-clean:
	rm -rf $(EXAMPLES)



