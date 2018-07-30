#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>

#define _PI 3.14159

class Simulator {
  private:
    float mf_alpha, mf_beta, mf_gamma; 
    float mf_trans_x, mf_trans_y, mf_trans_z; 
    cv::Mat mm_image;
    cv::Mat mm_Kf;
    cv::Mat mm_H;
    
    const cv::Size m_canvas_size;

    const float mfTRANS_PARAM = 0.001;
    const float mfROT_PARAM = 0.1;

  public:
    Simulator(const cv::Mat& image);
    cv::Mat GenerateWarpedImage(int tx, int ty, int tz, int rx, int ry, int rz);
    cv::Mat GetWarp();
};
