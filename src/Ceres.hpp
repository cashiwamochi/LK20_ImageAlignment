#include <iostream>
#include <vector>
#include <string>

#include "ceres/ceres.h"
#include "glog/logging.h"

#include <opencv2/opencv.hpp>

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;
using ceres::CauchyLoss;

namespace LK20 {
  struct HomographyPhotometricCostFunction{
    HomographyPhotometricCostFunction(cv::Mat m_ref_image, cv::Mat m_cur_image){
      mm_ref_image = m_ref_image.clone();
      mm_cur_image = m_cur_image.clone();
    }
    
    template<typename T>
    bool operator () (const T* const h, T* residual) const {
      cv::Mat H = (cv::Mat_<T>(3,3) << h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], T(1.0));

      residual[0] = projected_point_on_im2[0]/projected_point_on_im2[2] - T(dst_point.x);
      return true;
    }
    cv::Mat mm_cur_image, mm_ref_image;
} // namespace
