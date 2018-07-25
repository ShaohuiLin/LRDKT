LRDKT in Caffe
==================

Copy all file into your Caffe folder, and modify your ``src/caffe/proto/caffe.proto`` file in ``LayerParameter``:

```
optional SoftmaxWithTParameter softmax_witht_param = 150;
```

Then add the following to the bottom ``caffe.proto``:

```
message SoftmaxWithTParameter {
  enum Engine {
    DEFAULT = 0;
    CAFFE = 1;
    CUDNN = 2;
  }
  optional Engine engine = 1 [default = DEFAULT];

  // The axis along which to perform the softmax -- may be negative to index
  // from the end (e.g., -1 for the last axis).
  // Any other axes will be evaluated as independent softmaxes.
  optional int32 axis = 2 [default = 1];
  optional float temp = 3 [default = 1];
}
```
