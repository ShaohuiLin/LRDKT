# LRDKT

Pretrained caffe model of LRDKT.

## Models
### AlexNet
227x227 center crop validation accuracy on ImageNet, tested on one GTX TITAN X GPU with batch_size=32.

 | Model | #Param. | #FLOPs | CPU speedup | GPU speedup | Top-1 Err. | Top-5 Err. |  
| ------------- | ------------- | ------------- |  ------------- |  ------------- |  ------------- |   ------------- | 
| [LRDKT-0.7](https://drive.google.com/open?id=19VIhI3YH6w1F5IQ9B84ltkUczDR6H5eg) | 18.7M | 0.27B | 2.04× | 1.8× | 40.89% | 18.22% | 
| [LRDKT-GAP](https://drive.google.com/open?id=1IppCOGaVbm1YxvNyHZSLUU4RayXDNp0N) | 1.1M | 0.26B | 2.22× | 2.0× | 45.28% | 21.88% | 


### VGG16
224x224 center crop validation accuracy on ImageNet, tested on one GTX TITAN X GPU with batch_size=32.

 | Model | #Param. | #FLOPs | CPU speedup | GPU speedup | Top-1 Err. | Top-5 Err. |  
| ------------- | ------------- | ------------- |  ------------- |  ------------- |  ------------- |   ------------- | 
| [LRDKT-0.7](https://drive.google.com/open?id=1qAJK-LK48z61anzO8ij_txT0G_6CGQze) | 30.5M | 2.43B | 2.43× | 2.27× | 31.16% | 10.84% | 
| [LRDKT-0.5](https://drive.google.com/open?id=1f_QWBXlND9FbXoMwNCCbAGOkaXpuFF2V) | 9.5M | 1.31B | 2.61× | 2.55× | 35.77% | 13.9% | 
| [LRDKT-GAP](https://drive.google.com/open?id=1CRlYBcOKQydVM7itrNgqhHdh7YNx63gG) | 3.3M | 2.41B | 2.46× | 2.33× | 31.84% | 11.43% | 
