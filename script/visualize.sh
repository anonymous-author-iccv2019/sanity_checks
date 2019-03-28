archs=(resnet50 resnet101)
methods=(vanilla loss guided smooth integrated)
use_gpu=true

for arch in ${archs[@]}; do
	for method in ${methods[@]}; do
		if $use_gpu ; then
			# randomly initialized model
			python ../src/gradients.py -a ${arch} -o ../logs/${arch}/${method}/RND -m ${method} --cuda
			# standard trained model
			python ../src/gradients.py -a ${arch} -o ../logs/${arch}/${method}/STD -m ${method} --cuda --pretrained
			# adversarially trained model
			python ../src/gradients.py -a ${arch} -o ../logs/${arch}/${method}/ADV -m ${method} --cuda --weight ../pretrained/adversarial/${arch}.pth
		else
			# randomly initialized model
			python ../src/gradients.py -a ${arch} -o ../logs/${arch}/${method}/RND -m ${method}
			# standard trained model
			python ../src/gradients.py -a ${arch} -o ../logs/${arch}/${method}/STD -m ${method} --pretrained
			# adversarially trained model
			python ../src/gradients.py -a ${arch} -o ../logs/${arch}/${method}/ADV -m ${method} --weight ../pretrained/adversarial/${arch}.pth
		fi
	done
done
