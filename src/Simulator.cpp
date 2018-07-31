#include "Simulator.hpp"
#include <cmath>

using namespace std;

Simulator::Simulator(const cv::Mat& image)
{
  mm_image = image.clone();

	mf_alpha = 0.f; mf_beta = 0.f; mf_gamma = 0.f;
	mf_trans_x = 0.f; mf_trans_y = 0.f; mf_trans_z = 1.f;

  cv::Mat K = (cv::Mat_<float>(3,3) << 1000.f, 0.f, (float)image.cols/2.f,
                                       0.f, 1000.f, (float)image.rows/2.f,
                                       0.f, 0.f, 1.f);
  mm_Kf = K.clone();
}

cv::Mat Simulator::GenerateWarpedImage(int tx, int ty, int tz, int rx, int ry, int rz) {
  cv::Mat m_warped_image;

  // Updata !
  {
    mf_trans_x += mfTRANS_PARAM * (float)tx;
    mf_trans_y += mfTRANS_PARAM * (float)ty;
    mf_trans_z += mfTRANS_PARAM * (float)tz;

    mf_alpha += mfROT_PARAM/180.0*M_PI * (float)rx;
    mf_beta += mfROT_PARAM/180.0*M_PI * (float)ry;
    mf_gamma += mfROT_PARAM/180.0*M_PI * (float)rz;
  }

  cv::Mat mat_rot = cv::Mat::zeros(3,3,CV_32FC1);
  cv::Mat mat_H = cv::Mat::zeros(3,3,CV_32FC1);

  cv::Mat mat_rot_x = (cv::Mat_<float>(3,3) << 1.f, 0.f, 0.f,
                                               0.f, cosf(mf_alpha),sinf(mf_alpha),
                                               0.f, -sinf(mf_alpha),cosf(mf_alpha));

  cv::Mat mat_rot_y = (cv::Mat_<float>(3,3) << cosf(mf_beta), 0.f, -sinf(mf_beta),
                                               0.f, 1.f, 0.f,
                                               sinf(mf_beta), 0.f, cosf(mf_beta));

  cv::Mat mat_rot_z = (cv::Mat_<float>(3,3) << cosf(mf_gamma), sinf(mf_gamma), 0.f,
                                              -sinf(mf_gamma), cosf(mf_gamma), 0.f,
                                              0.f, 0.f, 1.f);

	mat_rot = mat_rot_z * mat_rot_y * mat_rot_x;

  cv::Mat vec_t = (cv::Mat_<float>(3,1) << mf_trans_x, mf_trans_y, mf_trans_z);

  mat_H.at<float>(0,0) = mat_rot.at<float>(0,0);
  mat_H.at<float>(1,0) = mat_rot.at<float>(1,0);
  mat_H.at<float>(2,0) = mat_rot.at<float>(2,0);
  mat_H.at<float>(0,1) = mat_rot.at<float>(0,1);
  mat_H.at<float>(1,1) = mat_rot.at<float>(1,1);
  mat_H.at<float>(2,1) = mat_rot.at<float>(2,1);
  mat_H.at<float>(0,2) = vec_t.at<float>(0,0);
  mat_H.at<float>(1,2) = vec_t.at<float>(1,0);
  mat_H.at<float>(2,2) = vec_t.at<float>(2,0);

	mat_H = mm_Kf * mat_H.clone() * mm_Kf.inv();
	mat_H = mat_H.clone()/(mat_H.clone()).at<float>(2,2);

	cv::warpPerspective( mm_image, m_warped_image, mat_H,
                       mm_image.size(),
                       cv::INTER_LINEAR,
                       cv::BORDER_CONSTANT, cv::Scalar(0,0,0) );

  mm_H = mat_H.clone();
  return m_warped_image;
}

cv::Mat Simulator::GetWarp() {
  return mm_H;
}
