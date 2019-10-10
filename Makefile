

SUBDIRS=examples tests

default: setup

all: setup test examples

.PHONY: test
test:
	$(MAKE) -C tests

.PHONY: examples
examples:
	$(MAKE) -C examples

.PHONY: setup
setup: googletest

googletest:
	rm -rf googletest
	git clone https://github.com/google/googletest.git
	cd googletest;	cmake CMakeLists.txt; make

cse141pp-archlab/compile.make:
	rm -rf cse141pp-archlab
	git clone https://github.com/NVSL/cse141pp-archlab.git


.PHONY: clean
clean:
	for i in $(SUBDIRS); do $(MAKE) -C $$i clean;done

