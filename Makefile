SUBDIRS=datasets/mnist datasets/cifar datasets/imagenet tests examples tools 


all: cse141pp-archlab examples tools datasets test
setup: #all tidy

.PHONY: test
test: check_env
	$(MAKE) -C tests

.PHONY: disttest
disttest:
	rm -rf ./.test_build
	git clone . ./.test_build
	bash -c '(cd .test_build; make setup; . ./env.sh; make C_OPTS=-O3 TESTS=*)'

.PHONY: examples
examples: check_env
	$(MAKE) -C examples

.PHONY: tools
tools: check_env
	$(MAKE) -C tools

.PHONY: datasets
datasets: check_env tools		
	for i in $(SUBDIRS); do $(MAKE) -C $$i; done

.PHONY: setup
setup: 

.PHONY: check_env
check_env:
	@if [ "$(ARCHLAB_ROOT)." = "." ]; then echo You need to do 'source ./env.sh' to setup your environment.; false; else true; fi


cse141pp-archlab:
	rm -rf cse141pp-archlab
	git clone https://github.com/NVSL/cse141pp-archlab.git

.PHONY: hooks
hooks:
	git config core.hooksPath hooks

.PHONY: clean
clean:
	rm -rf .test_build
	for i in $(SUBDIRS); do $(MAKE) -C $$i clean;done

.PHONY: tidy
tidy:
	rm -rf .test_build
	for i in $(SUBDIRS); do $(MAKE) -C $$i tidy;done

.PHONY: distclean
distclean:
	rm -rf googletest cse141pp-archlab
