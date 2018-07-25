#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
__global__ void CrossEntropyLossForwardGPU(const int count, const int dim,
          const Dtype* input_data, const Dtype* target, Dtype* loss) {
  CUDA_KERNEL_LOOP(i, count) {
		loss[i] = -target[i] * log(max(input_data[i], Dtype(FLT_MIN)));
    }
  }

template <typename Dtype>
__global__ void CrossEntropyLossBackwardGPU(const int count,
    const Dtype* input_data, const Dtype* target, Dtype* diff) {
  CUDA_KERNEL_LOOP(i, count) {
    diff[i] = -target[i] / max(input_data[i], Dtype(FLT_MIN));
  }
}


template <typename Dtype>
void CrossEntropyLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* input_data = bottom[0]->gpu_data();
  const Dtype* target = bottom[1]->gpu_data();
  const int count = bottom[0]->count();
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  
  CrossEntropyLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>(count, inner_num_, input_data, target, loss_data);
  
  Dtype loss;
  caffe_gpu_asum(count, loss_data, &loss);
 
  top[0]->mutable_cpu_data()[0] = loss / outer_num_;

  // Clear scratch memory to prevent interfering with backward.
  caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_gpu_diff());
  caffe_gpu_set(bottom[1]->count(), Dtype(0), bottom[1]->mutable_gpu_diff());
}

template <typename Dtype>
void CrossEntropyLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
   
    const int count = bottom[0]->count();
    const Dtype* input_data = bottom[0]->gpu_data();
    const Dtype* target = bottom[1]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    
    CrossEntropyLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>(count, input_data, target, bottom_diff);

    Dtype loss_weight = top[0]->cpu_diff()[0] / outer_num_;
    caffe_gpu_scal(count, loss_weight, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CrossEntropyLossLayer);

}  // namespace caffe
