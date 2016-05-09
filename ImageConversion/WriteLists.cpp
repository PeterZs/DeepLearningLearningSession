#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <fstream>
#include <iostream>
#include <set>

#include <gflags/gflags.h>

DEFINE_int32(num_images, 10000, "number of images");
DEFINE_double(testing_image_ratio, 0.05, "ratio of testing images");
DEFINE_string(image_path_1, "", "image path 1");
DEFINE_string(image_path_2, "", "image path 2");
DEFINE_string(image_path_3, "", "image path 3");
DEFINE_string(output_path, "", "output path");

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
  google::ParseCommandLineFlags(&argc, &argv, true);
  
  set<int> invalid_images_1;
  // invalid_images_1.insert(7019);
  // invalid_images_1.insert(7110);
  // invalid_images_1.insert(7177);
  // invalid_images_1.insert(7178);
  // invalid_images_1.insert(7208);
  // invalid_images_1.insert(7219);
  // invalid_images_1.insert(7286);
  // invalid_images_1.insert(7334);
  // invalid_images_1.insert(7357);
  // invalid_images_1.insert(9180);
  // invalid_images_1.insert(9489);
  set<int> invalid_images_2;
  // invalid_images_2.insert(9099);
  // invalid_images_2.insert(7348);
  // invalid_images_2.insert(7269);
  // invalid_images_2.insert(7168);
  // invalid_images_2.insert(7160);
  // invalid_images_2.insert(7117);
  // invalid_images_2.insert(7063);
  // invalid_images_2.insert(2100);
  ofstream train_list_out_str(FLAGS_output_path + "/train.txt");
  ofstream test_list_out_str(FLAGS_output_path + "/test.txt");
  for (int image_index = 1; image_index <= FLAGS_num_images; image_index++) {
    string image_filename_1 = FLAGS_image_path_1 + "/" + to_string(image_index) + ".png";
    string image_filename_2 = FLAGS_image_path_2 + "/" + to_string(image_index) + ".png";
    //string image_filename_3 = FLAGS_image_path_3 + "/" + to_string(image_index) + ".png";
    if (invalid_images_1.count(image_index) == 0) {
      if (rand() % 10000 < FLAGS_testing_image_ratio * 10000)
        test_list_out_str << image_filename_1 << " 0" << endl;
      else
        train_list_out_str << image_filename_1 << " 0" << endl;
    }
    if (invalid_images_2.count(image_index) == 0) {
      if (rand() % 10000 < FLAGS_testing_image_ratio * 10000)
	test_list_out_str << image_filename_2 << " 1" << endl;
      else
	train_list_out_str << image_filename_2 << " 1" << endl;
    }
    // if (rand() % 10000 < FLAGS_testing_image_ratio * 10000)
    //   test_list_out_str << image_filename_3 << " 2" << endl;
    // else
    //   train_list_out_str << image_filename_3 << " 2" << endl;
  }
  train_list_out_str.close();
  test_list_out_str.close();
  return 0;
}
