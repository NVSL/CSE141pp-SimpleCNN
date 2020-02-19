default: imagenet.dataset mininet.dataset

imagenet.dataset: image-list.txt
	../../tools/build_imagenet.exe 224 $@

mininet.dataset: image-list.txt
	../../tools/build_imagenet.exe 112 $@

image-list.txt:
	ls *.jpg > image-list.txt || rm image-list.txt

.PHONY:imagenet-clean		
imagenet-clean:
	rm -rf imagenet.dataset mininet.dataset image-list.txt

clean:imagenet-clean
