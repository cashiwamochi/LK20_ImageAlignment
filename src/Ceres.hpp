#include <iostream>
#include <vector>
#include <string>

#include "ceres/ceres.h"
#include "glog/logging.h"

#include <opencv2/opencv.hpp>

using ceres::AutoDiffCostFunction;
using ceres::NumericDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;
using ceres::CauchyLoss;

namespace LK20 {
  struct HomographyPhotometricCostFunction2{
    HomographyPhotometricCostFunction2(cv::Mat m_ref_image, cv::Mat m_cur_image, cv::Mat m_H0){
      m_ref_image.convertTo(mm_ref_image,CV_64F);
      m_cur_image.convertTo(mm_cur_image,CV_64F);
      m_H0.convertTo(mm_H0, CV_64F);
    }
    
    bool operator () (const double* const h, double* residual) const {
      cv::Mat _H = (cv::Mat_<double>(3,3) <<(h[0]), (h[1]), (h[2]),
                                            (h[3]), (h[4]), (h[5]),
                                            (h[6]), (h[7]), (1.0));

      cv::Mat m_dst_image;
      cv::warpPerspective(mm_cur_image, m_dst_image, mm_H0*_H,
                          cv::Size(mm_ref_image.cols, mm_ref_image.rows),
                          cv::INTER_LINEAR + cv::WARP_INVERSE_MAP, cv::BORDER_CONSTANT,
                          cv::Scalar(0,0,0));

      double r = 0.0;
      for (int i = 0; i < mm_ref_image.rows; i++) {
        for (int j = 0; j < mm_ref_image.cols; j++) {
          r -= (mm_ref_image.at<double>(i,j) - m_dst_image.at<double>(i,j));
        }
      }

      residual[0] = r/(double)(mm_ref_image.rows*mm_cur_image.cols);
      return true;
    }
    cv::Mat mm_cur_image, mm_ref_image;
    cv::Mat mm_H0;
  };
} // namespace

