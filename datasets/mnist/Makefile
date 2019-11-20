
default: all

all: emnist mnist

.PHONY: mnist emnist
mnist: mnist-test.dataset mnist-train.dataset

emnist: emnist-byclass-test.dataset emnist-byclass-train.dataset

## MNIST

mnist-test.dataset:  t10k-images-idx3-ubyte t10k-labels-idx1-ubyte
	../../tools/build_mnist.exe $^ $@ digits
mnist-train.dataset: train-images-idx3-ubyte train-labels-idx1-ubyte
	../../tools/build_mnist.exe $^ $@ digits

%-ubyte:
	curl -L http://yann.lecun.com/exdb/mnist/$*-ubyte.gz -o $*-ubyte.gz
	gunzip $*-ubyte.gz

## EMNIST

EMNIST_INPUTS=gzip/emnist-byclass-train-images-idx3-ubyte gzip/emnist-byclass-test-images-idx3-ubyte gzip/emnist-byclass-train-labels-idx1-ubyte gzip/emnist-byclass-test-labels-idx1-ubyte

gzip.zip:
# this url is for convenience.  If it breaks, the original is here:http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip
	curl -L https://storage.googleapis.com/nvsl_misc/gzip.zip -o $@

$(EMNIST_INPUTS): gzip.zip
	unzip -o $<
	gunzip -f gzip/*.gz
	touch gzip/*

emnist-byclass-test.dataset:  gzip/emnist-byclass-test-images-idx3-ubyte gzip/emnist-byclass-test-labels-idx1-ubyte
	../../tools/build_mnist.exe $^ $@ extended

emnist-byclass-train.dataset: gzip/emnist-byclass-train-images-idx3-ubyte gzip/emnist-byclass-train-labels-idx1-ubyte
	../../tools/build_mnist.exe $^ $@ extended


.PHONY:mnist-clean		
mnist-clean:
	rm -rf gzip.zip *.dataset *-ubyte *.gz gzip
clean:mnist-clean

