default all:
	$(CXX) -g -pg Example\ MNIST/example1.cpp -I . -o example1 -O3
	#	$(CXX) -g -pg Example\ MNIST/example2.cpp -I . -o example2 -O3
	$(CXX) -g -pg util/dump_mnist.cpp -I . -o dump_mnist -O3

.PHONY: setup
setup:
	rm -rf googletest
	git clone https://github.com/google/googletest.git
	cd googletest;	cmake CMakeLists.txt; make

