#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

namespace LK20 {

  enum CalcType {
    FC = 0, // Forward Compositional
    IC = 1, // Inverse Compositional
    ESM = 2, // Efficient Second-Order Mimization
  };

  enum ParamType {
    SL3 = 0, // H is parametrized by SL3
    SE3 = 1, // H is parametrtized by SE3
  };

  class LKTracker {
    public:
      LKTracker(cv::Mat image, cv::Rect rect, int pyramid_level = 4, CalcType t0 = ESM, ParamType t1 = SL3);
      ~LKTracker() { if(mb_verbose) { cv::destroyWindow(ms_window_name); }};
      void SetInitialWarp(const cv::Mat _H0);
      void Track(const cv::Mat m_target_image, cv::Mat& m_H, cv::Mat& m_dst_image);
      void SetVerbose(bool b_verbose);
      void SetTrueWarp(const cv::Mat& m_H_true);
      void Spin();

    private:
      void PreCompute();
      void RegisterSL3();
      void RegisterSE3();

      void ComputeJwJg();
      cv::Mat ComputeJ(const cv::Mat& m_dxdy, const cv::Mat& m_ref_dxdy = cv::noArray().getMat());
      cv::Mat ComputeHessian(const cv::Mat& m_j);

      cv::Mat ComputeImageGradient(const cv::Mat& _image);
      float ComputeResiduals(cv::Mat& m_redisuals, 
                             const cv::Mat& cur_image, 
                             const cv::Mat& ref_image);
      cv::Mat ComputeUpdateParams(const cv::Mat& m_hessian, 
                                  const cv::Mat& m_J,
                                  const cv::Mat& m_residuals);

      cv::Mat ProcessInLayer(const cv::Mat m_tmp_H,
                             const cv::Mat m_cur_image_in_pyramid,
                             const cv::Mat m_ref_image_in_pyramid,
                             const int level);

      cv::Mat WarpCurrentImage(const cv::Mat& src, const cv::Mat& H);

      void UpdateWarp(const cv::Mat& m_params, cv::Mat& new_H);

      bool IsConverged(const float& current_error, const float& previous_error);

      std::vector<cv::Mat> GenerateImagePyramid(const cv::Mat& m_src, const int num_of_levels);

      void ShowProcess(cv::Mat m_canvas, const cv::Mat& m_H);

      cv::Mat mm_current_original_image;
      const cv::Mat mm_ref_image; // This is an original (maybe colored) image for viewer
      cv::Mat mm_ref_working_image; // gray-scale

      std::vector<cv::Mat> mvm_ref_image_pyramid;

      const int m_pyramid_level;
      const CalcType m_type;
      const ParamType m_param_type;

      const int m_pyramid_factor = 2;
      const int m_iter_max = 100;

      std::vector<cv::Mat> mvm_SL3_bases;
      cv::Mat mm_Jg;
      std::vector<cv::Mat> mvm_JwJg;

      std::vector<cv::Mat> mvm_SE3_bases;


      // The length is m_pyramid_level
      std::vector<cv::Mat> mvm_ref_DxDy;
      std::vector<cv::Mat> mvm_J;
      std::vector<cv::Mat> mvm_hessian;

      cv::Mat mmK;
      cv::Mat mm_H0;
      cv::Mat mm_H_gt;

      const int m_height, m_width;

      bool mb_verbose;
      const std::string ms_window_name = "LK20 Tracker ( Press q to exit )";
  };
}
