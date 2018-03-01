# LRDKT

Pretrained caffe model of LRDKT.

## Models
224x224 center crop validation accuracy on ImageNet, tested on one GTX TITAN X GPU with batch_size=50.

| Model |#Param. | #FLOPs | CPU speedup | GPU speedup | Top-1 Err. | Top-5 Err. |
|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|
| VGG16 | 138.63M | 15.62B | 1.00× | 1.00× | 31.66% | 11.55%|
| LRDKT-0.7 | 30.5M | 2.43B | 2.43× | 2.27× | 31.16% | 10.84% |
| LRDKT-0.5 | 9.5M | | 2.61× | 2.55× | 35.77% | 13.9% |
| LRDKT+GAP | 3.3M | | 2.46× | 2.33× | 31.84% | 11.43% |
