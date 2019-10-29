default: all

all: cifar10 cifar100

cifar10: cifar10_data_batch_1.dataset cifar10_data_batch_2.dataset cifar10_data_batch_3.dataset cifar10_data_batch_4.dataset cifar10_data_batch_5.dataset cifar10_test_batch.dataset

.PRECIOUS: cifar-100-binary/%.bin
cifar-100-binary/%.bin:
	curl https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz -o  cifar-100-binary.tar.gz 
	tar xzf cifar-100-binary.tar.gz

cifar100: cifar100_test.dataset cifar100_train.dataset
cifar100_%.dataset: cifar-100-binary/%.bin
	../../tools/build_cifar.exe $^ 100 $@


.PRECIOUS: cifar-10-batches-bin/%.bin
cifar-10-batches-bin/%.bin:
	curl https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz -o  cifar-10-binary.tar.gz
	tar xzf cifar-10-binary.tar.gz

cifar10_%.dataset: cifar-10-batches-bin/%.bin
	../../tools/build_cifar.exe $^ 10 $@



.PHONY:cifar-clean		
cifar-clean:
	rm -rf make *.dataset *.tar.gz cifar-100-binary cifar-10-batches-bin
clean:cifar-clean

