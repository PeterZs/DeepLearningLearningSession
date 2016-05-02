#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/multi_image_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

  template <typename Dtype>
  MultiImageDataLayer<Dtype>::~MultiImageDataLayer<Dtype>() {
    this->StopInternalThread();
  }

  template <typename Dtype>
  void MultiImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
    const int new_height = this->layer_param_.multi_image_data_param().new_height();
    const int new_width  = this->layer_param_.multi_image_data_param().new_width();
    const bool is_color  = this->layer_param_.multi_image_data_param().is_color();
    string root_folder = this->layer_param_.multi_image_data_param().root_folder();
    const int num_images = this->layer_param_.multi_image_data_param().num_images();
    
    CHECK((new_height == 0 && new_width == 0) ||
          (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
    CHECK_GT(num_images, 0) << "The number of images should be positive.";
    // Read the file with filenames and labels
    const string& source = this->layer_param_.multi_image_data_param().source();
    LOG(INFO) << "Opening file " << source;
    std::ifstream infile(source.c_str());
    std::string line;
    while (std::getline(infile, line)) {
      std::istringstream iss(line);
      std::vector<string> filenames(num_images);
      for (int image_index = 0; image_index < num_images; image_index++)
        iss >> filenames[image_index];
      int label;
      iss >> label;
      lines_.push_back(std::make_pair(filenames, label));
    }

    if (this->layer_param_.multi_image_data_param().shuffle()) {
      // randomly shuffle data
      LOG(INFO) << "Shuffling data";
      const unsigned int prefetch_rng_seed = caffe_rng_rand();
      prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
      ShuffleImages();
    }
    LOG(INFO) << "A total of " << lines_.size() << " x " << num_images << " images.";

    lines_id_ = 0;
    // Check if we would need to randomly skip a few data points
    if (this->layer_param_.multi_image_data_param().rand_skip()) {
      unsigned int skip = caffe_rng_rand() %
        this->layer_param_.multi_image_data_param().rand_skip();
      LOG(INFO) << "Skipping first " << skip << " data points.";
      CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
      lines_id_ = skip;
    }
    // Read an image, and use it to initialize the top blob.
    CHECK(lines_[lines_id_].first.size()) << "There is no image in the first line.";
    cv::Mat cv_img = ReadImageToCVMat(root_folder + *lines_[lines_id_].first.begin(),
                                      new_height, new_width, is_color);
    CHECK(cv_img.data) << "Could not load " << *lines_[lines_id_].first.begin();
    // Use data_transformer to infer the expected blob shape from a cv_image.
    vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
    this->transformed_data_.Reshape(top_shape);
    top_shape[1] *= num_images;
    // Reshape prefetch_data and top[0] according to the batch_size.
    const int batch_size = this->layer_param_.multi_image_data_param().batch_size();
    CHECK_GT(batch_size, 0) << "Positive batch size required";
    top_shape[0] = batch_size;
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].data_.Reshape(top_shape);
    }
    top[0]->Reshape(top_shape);

    LOG(INFO) << "output data size: " << top[0]->num() << ","
              << top[0]->channels() << "," << top[0]->height() << ","
              << top[0]->width();
    // label
    vector<int> label_shape(1, batch_size);
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(label_shape);
    }
  }

  template <typename Dtype>
  void MultiImageDataLayer<Dtype>::ShuffleImages() {
    caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
    shuffle(lines_.begin(), lines_.end(), prefetch_rng);
  }

  // This function is called on prefetch thread
  template <typename Dtype>
  void MultiImageDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
    CPUTimer batch_timer;
    batch_timer.Start();
    double read_time = 0;
    double trans_time = 0;
    CPUTimer timer;
    CHECK(batch->data_.count());
    CHECK(this->transformed_data_.count());
    MultiImageDataParameter multi_image_data_param = this->layer_param_.multi_image_data_param();
    const int batch_size = multi_image_data_param.batch_size();
    const int new_height = multi_image_data_param.new_height();
    const int new_width = multi_image_data_param.new_width();
    const bool is_color = multi_image_data_param.is_color();
    string root_folder = multi_image_data_param.root_folder();
    const int num_images = this->layer_param_.multi_image_data_param().num_images();
    
    // Reshape according to the first image of each batch
    // on single input batches allows for inputs of varying dimension.
    cv::Mat cv_img = ReadImageToCVMat(root_folder + *lines_[lines_id_].first.begin(),
                                      new_height, new_width, is_color);
    CHECK(cv_img.data) << "Could not load " << *lines_[lines_id_].first.begin();
    // Use data_transformer to infer the expected blob shape from a cv_img.
    vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
    this->transformed_data_.Reshape(top_shape);
    top_shape[1] *= num_images;
    // Reshape batch according to the batch_size.
    top_shape[0] = batch_size;
    batch->data_.Reshape(top_shape);

    Dtype* prefetch_data = batch->data_.mutable_cpu_data();
    Dtype* prefetch_label = batch->label_.mutable_cpu_data();

    // datum scales
    const int lines_size = lines_.size();
    for (int item_id = 0; item_id < batch_size; ++item_id) {
      // get a blob
      timer.Start();
      CHECK_GT(lines_size, lines_id_);
      
      if (this->layer_param_.multi_image_data_param().shuffle_images() == true) {
	caffe::rng_t* prefetch_rng =
	  static_cast<caffe::rng_t*>(prefetch_rng_->generator());
	shuffle(lines_[lines_id_].first.begin(), lines_[lines_id_].first.end(), prefetch_rng);
      }
      read_time += timer.MicroSeconds();
      timer.Start();
      for (int image_index = 0; image_index < num_images; image_index++) {
             cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first[image_index], new_height, new_width, is_color);
             CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first[image_index];
             // Apply transformations (mirror, crop...) to the image
             int offset = batch->data_.offset(item_id, image_index * cv_img.channels());
             this->transformed_data_.set_cpu_data(prefetch_data + offset);
             this->data_transformer_->Transform(cv_img, &(this->transformed_data_));	     
      }

      trans_time += timer.MicroSeconds();

      prefetch_label[item_id] = lines_[lines_id_].second;
      // go to the next iter
      lines_id_++;
      if (lines_id_ >= lines_size) {
        // We have reached the end. Restart from the first.
        DLOG(INFO) << "Restarting data prefetching from start.";
        lines_id_ = 0;
        if (this->layer_param_.multi_image_data_param().shuffle()) {
          ShuffleImages();
        }
      }
    }
    batch_timer.Stop();
    DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
    DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
    DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
  }

  INSTANTIATE_CLASS(MultiImageDataLayer);
  REGISTER_LAYER_CLASS(MultiImageData);

}  // namespace caffe
#endif  // USE_OPENCV
