include ../Make.rules

TESTS?=-*SLOW*

.PHONY:default test
default test: core_tests

core_tests:run_test.exe 
	./run_test.exe --gtest_filter=$(TESTS)

GCOV_CFLAGS=-fprofile-arcs -ftest-coverage

%.exe : %.o 
	$(CXX) $(USER_LDFLAGS) $(LD_OPTS) -L../googletest/lib/  $^ -lgtest -lgtest_main -lpng -ljpeg  -pthread -o $@

.PHONY: codecov
codecov:
	  for f in ../CNN/* run_test.cpp; do gcov -n -r -o ./run_test.exe $$f; done > /dev/null; 

.PHONY: tests-clean
clean: tests-clean
tests-clean:
	rm -rf run_test.exe

.PHONY: tests-help
help: tests-help
tests-help:
	@echo "make TESTS=<google_test test filter>  : just run some tests. (currently=$(TESTS))"

