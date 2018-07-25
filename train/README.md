LRDKT train
============

1. Download vgg16 caffemodel and prototxt from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md

2. Using decom.py decompose model
```
python decom.py --model VGG_ILSVRC_16_train_val.prototxt --weights VGG_ILSVRC_16_layers.caffemodel --outputmodel vgg16_0.7.prototxt --outputweights vgg16_0.7.caffemodel --pca 0.7
```

- `--model`： The original network prototxt
- `--weights`： The original network caffemodel
- `--outputmodel`： The decompose network prototxt
- `--outputweights`： The decompose network caffemodel
- `--pca`： decompose pca value

3. Using generate_model.py combine teacher network and student network
```
python generate_model.py --model VGG_ILSVRC_16_train_val.prototxt --weights VGG_ILSVRC_16_layers.caffemodel --input vgg16_0.7.caffemodel --output vgg16_KT_0.7.caffemodel
```

- `--model`： The teacher network prototxt
- `--weights`： The teacher network caffemodel
- `--input`： The student network caffemodel
- `--output`： The dir of output caffemodel

4. train
```
{The root directory of caffe}/build/tools/caffe train --solver solver_all_0.7_LRDKT.prototxt --weights vgg16_KT_0.7.caffemodel --gpu 0
```
