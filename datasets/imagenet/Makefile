imagenet.dataset: image-list.txt
	../../tools/build_imagenet.exe 

image-list.txt:
	ls *.jpg > image-list.txt || rm image-list.txt

.PHONY:imagenet-clean		
imagenet-clean:
	rm -rf imagenet.dataset image-list.txt

clean:imagenet-clean
