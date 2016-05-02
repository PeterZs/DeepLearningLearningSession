#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>


#include <gflags/gflags.h>

DEFINE_string(input_filename, "", "input filename");
DEFINE_string(output_path, "", "output path");
DEFINE_int32(num_images, 100, "the number of images to extract");

using namespace std;
using namespace cv;


int main(int argc, char *argv[])
{
  google::ParseCommandLineFlags(&argc, &argv, true);
  
  ifstream in_str(FLAGS_input_filename, ios::binary);
  if (!in_str)
    exit(1);
  int IMAGE_WIDTH = 32, IMAGE_HEIGHT = 32;
  ofstream out_str(FLAGS_output_path + "/list.txt");
  for (int image_index = 0; image_index < FLAGS_num_images; image_index++) {
    char label = 0;
    in_str.read(reinterpret_cast<char *>(&label), sizeof(char));
    vector<uchar> r_values(IMAGE_WIDTH * IMAGE_HEIGHT);
    in_str.read(reinterpret_cast<char *>(&r_values[0]), sizeof(char) * IMAGE_WIDTH * IMAGE_HEIGHT);
    vector<uchar> g_values(IMAGE_WIDTH * IMAGE_HEIGHT);
    in_str.read(reinterpret_cast<char *>(&g_values[0]), sizeof(char) * IMAGE_WIDTH * IMAGE_HEIGHT);
    vector<uchar> b_values(IMAGE_WIDTH * IMAGE_HEIGHT);
    in_str.read(reinterpret_cast<char *>(&b_values[0]), sizeof(char) * IMAGE_WIDTH * IMAGE_HEIGHT);
    
    string output_image_filename = FLAGS_output_path + "/image_" + to_string(image_index) + ".png";
    Mat image(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3);
    for (int pixel = 0; pixel < IMAGE_WIDTH * IMAGE_HEIGHT; pixel++)
      image.at<Vec3b>(pixel / IMAGE_WIDTH, pixel % IMAGE_WIDTH) = Vec3b(b_values[pixel], g_values[pixel], r_values[pixel]);
    imwrite(output_image_filename, image);
    
    out_str << output_image_filename << ' ' << static_cast<int>(label) << endl;
  }
  in_str.close();
  out_str.close();
  
  return 0;
}
