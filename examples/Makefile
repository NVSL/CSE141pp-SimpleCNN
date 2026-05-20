CXX=g++
USER_CFLAGS += -I$(GOOGLE_TEST_ROOT)/googletest/include/ -I..  -std=c++14 -O3
include ../Make.rules

EXAMPLES=alexnet.exe toy.exe simple.exe
default: $(EXAMPLES)

%.exe : %.o 
	$(CXX) $(GENERIC_LDFLAGS) $^ -o $@

tidy:

clean: examples-clean
examples-clean:
	rm -rf $(EXAMPLES)



