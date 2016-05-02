#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <fstream>
#include <iostream>

#include <gflags/gflags.h>

DEFINE_string(input_filename, "", "input filename");
DEFINE_string(output_path, "", "output path");

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
  google::ParseCommandLineFlags(&argc, &argv, true);
  
  ifstream in_str(FLAGS_input_filename);
  ofstream out_str(FLAGS_output_path + "/list.txt");
  string line;
  int index = 0;
  while (getline(in_str, line)) {
    stringstream ss(line);
    string input_image_filename;
    ss >> input_image_filename;
    Mat image = imread(input_image_filename);
    Mat label_image = Mat::zeros(image.size(), CV_8UC1);
    label_image.setTo(1);
    string output_image_filename = FLAGS_output_path + "/label_" + to_string(index) + ".png";
    imwrite(output_image_filename, label_image);
    index++;
    
    out_str << input_image_filename << ' ' << output_image_filename << endl;
  }
  in_str.close();
  out_str.close();
  return 0;
}
