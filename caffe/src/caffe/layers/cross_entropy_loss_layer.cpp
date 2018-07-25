#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CrossEntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  
}

template <typename Dtype>
void CrossEntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  outer_num_ = bottom[0]->shape(0);  // batch size
  inner_num_ = bottom[0]->count(1);  // instance size: |output| == |target|
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "CROSS_ENTROPY_LOSS layer inputs must have the same count.";
}

template <typename Dtype>
void CrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();

  Dtype loss = 0;
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; j++) {
        loss -= target[i * inner_num_ + j] * log(std::max(input_data[i * inner_num_ + j],
                           Dtype(FLT_MIN)));
    }
  }

  top[0]->mutable_cpu_data()[0] = loss / outer_num_;
}

template <typename Dtype>
void CrossEntropyLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // compute the diff
    const int count = bottom[0]->count();
    const Dtype* input_data = bottom[0]->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	caffe_copy(count, bottom[0]->cpu_data(), bottom_diff);
	
    for (int i = 0; i < outer_num_; ++i) {
      for (int j = 0; j < inner_num_; j++) {
          bottom_diff[i * inner_num_ + j] = std::max(input_data[i * inner_num_ + j], Dtype(FLT_MIN));
      }
    }

    caffe_div(count, target, bottom_diff, bottom_diff);
    caffe_cpu_scale(count, Dtype(-1), bottom_diff, bottom_diff);
    // Scale down gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] / outer_num_;
    caffe_scal(count, loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(CrossEntropyLossLayer);
#endif

INSTANTIATE_CLASS(CrossEntropyLossLayer);
REGISTER_LAYER_CLASS(CrossEntropyLoss);

}  // namespace caffe
