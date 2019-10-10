include ../Make.rules

.PHONY:default test
default test: run_test.exe
	./run_test.exe

run_test.exe: run_test.cpp
	$(CXX) $(CXXFLAGS) -fprofile-arcs -ftest-coverage -L../googletest/lib/ $< -lgtest -lgtest_main  -o $@

.PHONY: codecov
codecov:
	  for f in ../CNN/* run_test.cpp; do gcov -n -r -o ./run_test.exe $$f; done > /dev/null; 

.PHONY: tests-clean
tests-clean:
	rm -rf run_test.exe

clean: tests-clean

