

SUBDIRS=examples tests

default: setup

all: setup test examples

.PHONY: test
test: check_env
	$(MAKE) -C tests

.PHONY: disttest
disttest:
	rm -rf ./.test_build
	git clone . ./.test_build
	bash -c '(cd .test_build; make C_OPTS=-O3; . ./env.sh; make all C_OPTS=-O3 TESTS=*)'

.PHONY: examples
examples: check_env
	$(MAKE) -C examples

.PHONY: setup
setup: googletest cse141pp-archlab

.PHONY: check_env
check_env:
	@if [ "$(ARCHLAB_ROOT)." = "." ]; then echo You need to do 'source ./env.sh' to setup your environment.; false; else true; fi

googletest:
	rm -rf googletest
	git clone https://github.com/google/googletest.git
	cd googletest;	cmake CMakeLists.txt; make

cse141pp-archlab:
	rm -rf cse141pp-archlab
	git clone https://github.com/NVSL/cse141pp-archlab.git

.PHONY: hooks
hooks:
	git config core.hooksPath hooks

.PHONY: clean
clean:
	for i in $(SUBDIRS); do $(MAKE) -C $$i clean;done

.PHONY: distclean
distclean:
	rm -rf googletest cse141pp-archlab
