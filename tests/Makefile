include ../Make.rules

default test: run_test.exe
	./run_test.exe

run_test.exe: run_test.cpp
	$(CXX) $(CXXFLAGS) -L../googletest/lib/ $< -lgtest -lgtest_main  -o $@


.PHONY: tests-clean
tests-clean:
	rm -rf run_test.exe

clean: tests-clean

