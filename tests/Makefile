include ../Make.rules

-include $(SRCS:.cpp=.d)

SRCS=run_test.cpp

.PHONY:default test
default test: run_test.exe
	./run_test.exe

GCOV_CFLAGS=-fprofile-arcs -ftest-coverage
run_test.exe: run_test.cpp
	$(CXX) $(CXXFLAGS)   -L../googletest/lib/ $< -lgtest -lgtest_main -lpng  -o $@

.PHONY: codecov
codecov:
	  for f in ../CNN/* run_test.cpp; do gcov -n -r -o ./run_test.exe $$f; done > /dev/null; 

.PHONY: tests-clean
tests-clean:
	rm -rf run_test.exe

clean: tests-clean


