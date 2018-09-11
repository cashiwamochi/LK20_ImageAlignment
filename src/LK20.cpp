#include "LK20.hpp"

namespace LK20 {
  LKTracker::LKTracker(cv::Mat image, cv::Rect rect, int pyramid_level, CalcType t) 
    : mm_ref_image(image), m_pyramid_level(pyramid_level), m_type(t), m_height(rect.height), m_width(rect.width)
  {  
    mm_H0.release(); // This ensures mm_H0 is empty.
    mb_verbose = false;
   // How to Calculate
    std::cout << "[MODE] : ";
    if(m_type == ESM) {
      std::cout << "Efficient Second-Order Minimization\n";
    }
    else if(m_type == IC) {
      std::cout << "Inverse Compositional\n";
    }
    else if(m_type == FC) {
      std::cout << "Forward Compositional\n";
    }

    // Preprocess
    if(mm_ref_image.channels() == 3) {
      cv::cvtColor(mm_ref_image, mm_ref_working_image, cv::COLOR_BGR2GRAY);
    }
    else {
      mm_ref_working_image = mm_ref_image.clone();
    }
    cv::GaussianBlur(mm_ref_working_image, mm_ref_working_image, cv::Size(5,5), 0);

    // Image Pyramid
    mvm_ref_image_pyramid.reserve(m_pyramid_level);
    std::vector<cv::Mat> vm_image_pyramid = GenerateImagePyramid(mm_ref_working_image, m_pyramid_level);
    for (int l = 0; l < m_pyramid_level; l++) {
      mvm_ref_image_pyramid.push_back(vm_image_pyramid[l](rect));
    }

    RegisterSL3();
    PreCompute();
  }

  cv::Mat LKTracker::ComputeImageGradient(const cv::Mat& _image) {
    cv::Mat m_dxdy = cv::Mat::zeros(m_height*m_width, 2, CV_32F);
    for(int v = 0; v < m_height; v++) {
      for(int u = 0; u < m_width; u++) {
        const int idx = u + v*m_width;
        float dx, dy;

        if( u+1 == m_width or v+1 == m_height ) {
          dx = 0.f;
          dy = 0.f;
        }
        else {
          dx = _image.at<float>(v,u+1) - _image.at<float>(v,u);
          dy = _image.at<float>(v+1,u) - _image.at<float>(v,u);
        }
        m_dxdy.at<float>(idx, 0) = dx;
        m_dxdy.at<float>(idx, 1) = dy;
      }
    }
    return m_dxdy;
  }

  cv::Mat LKTracker::ComputeHessian(const cv::Mat& m_j) {
    int num_of_params = (int)mvm_SL3_bases.size(); // 8
    cv::Mat m_hessian = cv::Mat::zeros(num_of_params, num_of_params, CV_32F);

    for(int r = 0; r < m_j.rows; r++) {
      m_hessian += m_j.row(r).t() * m_j.row(r);
    }
    // 8 x 8
    return m_hessian;
  }

  // Comute JiJwJg
  cv::Mat LKTracker::ComputeJ(const cv::Mat& m_dxdy, const cv::Mat& m_ref_dxdy) {
    cv::Mat J = cv::Mat::zeros(m_height*m_width, 8, CV_32F);

    cv::Mat m_Ji;
    if(m_type == ESM) {
      assert(!m_ref_dxdy.empty());
      m_Ji = (m_dxdy + m_ref_dxdy)/2.0;
    }

    for(int u = 0; u < m_height; u++) {
      for(int v = 0; v < m_width; v++) {
        int idx = u*m_width + v;
        cv::Mat _j;
        if(m_type == FC or m_type == IC) {
          _j = m_dxdy.row(idx) * mvm_JwJg[idx];
        }
        else if(m_type == ESM) {
          _j = m_Ji.row(idx) * mvm_JwJg[idx];          
        }
        _j.copyTo(J.row(idx));
      }
    }
    /*
    J = pixel-nuber x [1 x 8]
    */

    return J;
  }

  void LKTracker::ComputeJwJg() {
    // dW/dp (p=0,W=I)
    mvm_JwJg.clear();
    mvm_JwJg.reserve(m_height * m_width);
    for(int _v = 0; _v < m_height; _v++) {
      for(int _u = 0; _u < m_width; _u++) {
        float u = (float)_u;
        float v = (float)_v;

        /* PAPER p.12 (63) ただし，第3行成分は実際の計算では0で意味がないので削った */
        cv::Mat _mJw = (cv::Mat_<float>(2,9) <<
        u, v, 1.f, 0.f, 0.f, 0.f, -u*u, -u*v, -u,
        0.f, 0.f, 0.f, u, v, 1.f, -u*v, -v*v, -v);

        cv::Mat mfjwjg = _mJw * mm_Jg;
        // [2x8] = [2x9] x [9x8]
        mvm_JwJg.push_back(mfjwjg);
        // pixel-number x [2 x 8]
      }
    }
    return;
  }

  float LKTracker::ComputeResiduals(cv::Mat& m_residuals,
                                    const cv::Mat& cur_image,
                                    const cv::Mat& ref_image) {
    float f_residual = 0.f;

    m_residuals = cv::Mat::zeros(m_height*m_width,1,CV_32F);
    for(int v = 0; v < m_height; v++) {
      for(int u = 0; u < m_width; u++) {
        const int idx = v*m_width + u;
        float r;
        if(m_type == IC or m_type == ESM) {
          r = cur_image.at<float>(v,u) - ref_image.at<float>(v,u);
        }
        else if(m_type == FC) {
          r = ref_image.at<float>(v,u) - cur_image.at<float>(v,u);
        }
        m_residuals.at<float>(idx,0) = r;
        f_residual += r*r;
      }
    }
    return sqrt(f_residual/(float)(m_height*m_width));
  }

  cv::Mat LKTracker::ComputeUpdateParams(const cv::Mat& m_hessian, 
                                         const cv::Mat& m_J,
                                         const cv::Mat& m_residuals)  {
    cv::Mat m_params = cv::Mat::zeros(8,1,CV_32F);
    cv::Mat m_hessian_inv = m_hessian.inv();
    for(int v = 0; v < m_height; v++) {
      for(int u = 0; u < m_width; u++) {
        const int idx = u + v*m_width;
        if(m_type == ESM) {
          // m_params += -1.0*m_hessian_inv * m_J.row(idx).t() * m_residuals.at<float>(idx,0);
          m_params += -1.0 * m_J.row(idx).t() * m_residuals.at<float>(idx,0);
        }
        else {
          // m_params += m_hessian_inv * m_J.row(idx).t() * m_residuals.at<float>(idx,0);
          m_params += m_J.row(idx).t() * m_residuals.at<float>(idx,0);
        }
        // [8*1] = [8x8] * [8*1] * [1]
      }
    }

    m_params = m_hessian_inv * m_params;
    return m_params;
  }

  std::vector<cv::Mat> LKTracker::GenerateImagePyramid(const cv::Mat& m_src, const int num_of_levels) {
    std::vector<cv::Mat> vm_image_pyramid;
    vm_image_pyramid.reserve(num_of_levels);
    cv::Mat m_down_sampled_image = m_src.clone();
    
    cv::Mat _m_src;
    m_src.convertTo(_m_src, CV_32FC1, 1.0/255.0);
    vm_image_pyramid.push_back(_m_src);

    for(int l = 0; l < num_of_levels-1; l++) {
      cv::pyrDown(m_down_sampled_image.clone(), m_down_sampled_image, 
                  cv::Size(m_down_sampled_image.cols/2, m_down_sampled_image.rows/2));
      cv::Mat m_up_sampled_image = m_down_sampled_image.clone();
      for(int m = 0; m < l+1; m++) {
        cv::pyrUp(m_up_sampled_image.clone(), m_up_sampled_image, 
                  cv::Size(m_up_sampled_image.cols*2, m_up_sampled_image.rows*2));
      }

      m_up_sampled_image.convertTo(m_up_sampled_image, CV_32FC1, 1.0/255.0);
      vm_image_pyramid.push_back(m_up_sampled_image);
    }
    
    if(false) {
      for(int l = 0; l < num_of_levels; l++) {
        // this works with CV_32F mat
        cv::imshow("debug", vm_image_pyramid[l]);
        cv::waitKey(0);
      }
    }
    return vm_image_pyramid;
  }

  bool LKTracker::IsConverged(const float& current_error, const float& previous_error) {
    bool b_stop_iteration = false;
    if(previous_error < 0.0) {
      return false;
    }
    if(previous_error <= current_error + 0.0000001) {
      b_stop_iteration = true;
    }
    return b_stop_iteration;
  }


  void LKTracker::PreCompute() {
    switch(m_type) {
      case FC : {
        ComputeJwJg();
      } break;
      case IC : 
      {
        ComputeJwJg();
        mvm_ref_DxDy.clear(); mvm_ref_DxDy.reserve(m_pyramid_level);
        mvm_J.clear(); mvm_J.reserve(m_pyramid_level);
        mvm_hessian.clear(); mvm_hessian.reserve(m_pyramid_level);

        for(int level = 0; level < m_pyramid_level; level++) {
          cv::Mat m_dxdy = ComputeImageGradient(mvm_ref_image_pyramid[level]);
          mvm_ref_DxDy.push_back(m_dxdy);
          cv::Mat m_J = ComputeJ(m_dxdy); // JiJwJg
          mvm_J.push_back(m_J);
          cv::Mat m_hessian = ComputeHessian(m_J);
          mvm_hessian.push_back(m_hessian);
        }
      } break;
      case ESM :
      {
        // almost same as IC
        ComputeJwJg();
        mvm_ref_DxDy.clear(); mvm_ref_DxDy.reserve(m_pyramid_level);

        for(int level = 0; level < m_pyramid_level; level++) {
          cv::Mat m_dxdy = ComputeImageGradient(mvm_ref_image_pyramid[level]);
          mvm_ref_DxDy.push_back(m_dxdy);
        }
      } break;
     }
  }

  cv::Mat LKTracker::ProcessInLayer(const cv::Mat m_tmp_H, 
                                    const cv::Mat m_cur_image_in_pyramid,
                                    const cv::Mat m_ref_image_in_pyramid,
                                    const int level) {
    float previous_error = -1.0;
    cv::Mat H = m_tmp_H.clone();

    switch(m_type) {
      case IC : {
        // Inverse Compositional Algorithm
        for(int itr = 0; itr < m_iter_max; itr++) {
          cv::Mat m_cur_working_image = WarpCurrentImage(m_cur_image_in_pyramid.clone(), H);
          cv::Mat m_residuals;
          float current_error = ComputeResiduals(m_residuals, m_cur_working_image, m_ref_image_in_pyramid);
          fprintf(stderr,"Itr[%d|%d]: %.6f\n",itr,level,current_error);
          cv::Mat m_update_params = ComputeUpdateParams(mvm_hessian[level], mvm_J[level], m_residuals);
          cv::Mat new_H = H.clone();
          UpdateWarp(m_update_params, new_H);
          
          if((int)previous_error < 0) {
            // decide whether iterative update should be done in this level or not
            m_cur_working_image = WarpCurrentImage(m_cur_image_in_pyramid, new_H);
            previous_error = current_error;
            current_error = ComputeResiduals(m_residuals, m_cur_working_image, m_ref_image_in_pyramid);
            if(IsConverged(current_error, previous_error)) {
              break;
            }
            else {
              H = new_H.clone();
              if(mb_verbose) {
                ShowProcess(mm_ref_image, H);
              }
              continue;
            }
          }
          
          if(!IsConverged(current_error, previous_error)) 
          {  
            previous_error = current_error;
            H = new_H.clone();
            if(mb_verbose) {
                ShowProcess(mm_ref_image, H);
            }
          }
          else {
            break;
          }
        }
      } break;
      case FC : {
        // Forward Compositional Algorithm
        for(int itr = 0; itr < m_iter_max; itr++) {
          cv::Mat m_cur_working_image = WarpCurrentImage(m_cur_image_in_pyramid, H);
          cv::Mat m_residuals;
          float current_error = ComputeResiduals(m_residuals, m_cur_working_image, m_ref_image_in_pyramid);
          fprintf(stderr,"Itr[%d|%d]: %.6f\n",itr, level, current_error);
          cv::Mat m_dxdy = ComputeImageGradient(m_cur_working_image);
          cv::Mat m_J = ComputeJ(m_dxdy); // JiJwJg
          cv::Mat m_hessian = ComputeHessian(m_J);
          cv::Mat m_update_params = ComputeUpdateParams(m_hessian, m_J, m_residuals);
          cv::Mat new_H = H.clone();
          UpdateWarp(m_update_params, new_H);

          if((int)previous_error < 0) {
            // decide whether iterative update should be done in this level or not
            m_cur_working_image = WarpCurrentImage(m_cur_image_in_pyramid, new_H);
            previous_error = current_error;
            current_error = ComputeResiduals(m_residuals, m_cur_working_image, m_ref_image_in_pyramid);
            if(IsConverged(current_error, previous_error)) {
              break;
            }
            else {
              H = new_H.clone();
              if(mb_verbose) {
                ShowProcess(mm_ref_image, H);
              }
              continue;
            }
          }
          
          if(!IsConverged(current_error, previous_error)) 
          {  
            previous_error = current_error;
            H = new_H.clone();
            if(mb_verbose) {
              ShowProcess(mm_ref_image, H);
            }
          }
          else {
            break;
          }
        }
      } break;
      case ESM : {
        // Efficient Second-Order Minimization Algorithm
        for(int itr = 0; itr < m_iter_max; itr++) {
          cv::Mat m_cur_working_image = WarpCurrentImage(m_cur_image_in_pyramid, H);
          cv::Mat m_residuals;
          float current_error = ComputeResiduals(m_residuals, m_cur_working_image, m_ref_image_in_pyramid);
          fprintf(stderr,"Itr[%d|%d]: %.6f\n",itr, level, current_error);
          cv::Mat m_dxdy = ComputeImageGradient(m_cur_working_image);
          cv::Mat m_J = ComputeJ(m_dxdy, mvm_ref_DxDy[level]); // JiJwJg
          cv::Mat m_hessian = ComputeHessian(m_J);
          cv::Mat m_update_params = ComputeUpdateParams(m_hessian, m_J, m_residuals);
          cv::Mat new_H = H.clone();
          UpdateWarp(m_update_params, new_H);

          if((int)previous_error < 0) {
            // decide whether iterative update should be done in this level or not
            m_cur_working_image = WarpCurrentImage(m_cur_image_in_pyramid, new_H);
            previous_error = current_error;
            current_error = ComputeResiduals(m_residuals, m_cur_working_image, m_ref_image_in_pyramid);
            if(IsConverged(current_error, previous_error)) {
              break;
            }
            else {
              H = new_H.clone();
              if(mb_verbose) {
                ShowProcess(mm_ref_image, H);
              }
              continue;
            }
          }
          
          if(!IsConverged(current_error, previous_error)) 
          {  
            previous_error = current_error;
            H = new_H.clone();
            if(mb_verbose) {
                ShowProcess(mm_ref_image, H);
            }
          }
          else {
            break;
          }
        }
      } break;
    }

    return H;
  }

  void LKTracker::RegisterSL3() {
    // Register SL3 bases
    mvm_SL3_bases.resize(8);
    mvm_SL3_bases[0] = (cv::Mat_<float>(3,3) << 0.0, 0.0, 1.0,
                                                0.0, 0.0, 0.0,
                                                0.0, 0.0, 0.0);
    mvm_SL3_bases[1] = (cv::Mat_<float>(3,3) << 0.0, 0.0, 0.0,
                                                0.0, 0.0, 1.0,
                                                0.0, 0.0, 0.0);
    mvm_SL3_bases[2] = (cv::Mat_<float>(3,3) << 0.0, 1.0, 0.0,
                                                0.0, 0.0, 0.0,
                                                0.0, 0.0, 0.0);
    mvm_SL3_bases[3] = (cv::Mat_<float>(3,3) << 0.0, 0.0, 0.0,
                                                1.0, 0.0, 0.0,
                                                0.0, 0.0, 0.0);
    mvm_SL3_bases[4] = (cv::Mat_<float>(3,3) << 1.0, 0.0, 0.0,
                                                0.0,-1.0, 0.0,
                                                0.0, 0.0, 0.0);
    mvm_SL3_bases[5] = (cv::Mat_<float>(3,3) << 0.0, 0.0, 0.0,
                                                0.0,-1.0, 0.0,
                                                0.0, 0.0, 1.0);
    mvm_SL3_bases[6] = (cv::Mat_<float>(3,3) << 0.0, 0.0, 0.0,
                                                0.0, 0.0, 0.0,
                                                1.0, 0.0, 0.0);
    mvm_SL3_bases[7] = (cv::Mat_<float>(3,3) << 0.0, 0.0, 0.0,
                                                0.0, 0.0, 0.0,
                                                0.0, 1.0, 0.0);

    /* PAPER p.12 (65) */
    // make Jg
    mm_Jg = cv::Mat::zeros(9, 8, CV_32FC1);
    for(int i = 0; i < 8; i++) {
      for(int j = 0; j < 3; j++) {
        for(int k = 0; k < 3; k++) {
          mm_Jg.at<float>(j*3 + k,i) = mvm_SL3_bases[i].at<float>(j,k);
        }
      }
    }
    return;
  }

  void LKTracker::SetInitialWarp(const cv::Mat _H0) {
    mm_H0 = _H0.clone();
    return;
  }

  void LKTracker::SetTrueWarp(const cv::Mat& m_H_true) {
    mm_H_gt = m_H_true.clone();
    return;
  }

  void LKTracker::SetVerbose(bool b_verbose) {
    if(mm_H_gt.empty() && b_verbose) {
      std::cout << "[WARN] : The True Homography is not given. Process is not shown." << std::endl;
      std::cout << "         Please use SetTrueWarp()." << std::endl;
      return;
    }
    mb_verbose = b_verbose;
    return;
  }

  void LKTracker::ShowProcess(cv::Mat m_canvas, const cv::Mat& m_H) {
    cv::Mat p[4], ref_p[4];
    p[0] = (cv::Mat_<float>(3,1) << 0.f, 0.f, 1.f);
    p[1] = (cv::Mat_<float>(3,1) << (float)m_width, 0.f, 1.f);
    p[2] = (cv::Mat_<float>(3,1) << (float)m_width, (float)m_height, 1.f);
    p[3] = (cv::Mat_<float>(3,1) << 0.f, (float)m_height, 1.f);
    
    for(int i = 0; i < 4; i++) {
      ref_p[i] = p[i].clone();
    }

    for(int i = 0; i < 4; i++) {
      ref_p[i] = mm_H0 * ref_p[i];
      ref_p[i] = ref_p[i] / ref_p[i].at<float>(2);
      if(!mm_H_gt.empty()) {
        p[i] = mm_H0 * m_H.inv() * mm_H0.inv()* mm_H_gt * mm_H0 * p[i];
        p[i] = p[i] / p[i].at<float>(2);
      }
    }

    const std::vector<int> rect_idx{0,1,2,3,0};
    
    cv::Mat mat_for_viewer = m_canvas.clone();
    for(int i = 0; i < 4; i++){
      cv::line(mat_for_viewer,
               cv::Point2f(ref_p[rect_idx[i]].at<float>(0), ref_p[rect_idx[i]].at<float>(1)),
               cv::Point2f(ref_p[rect_idx[i+1]].at<float>(0), ref_p[rect_idx[i+1]].at<float>(1)),
               cv::Scalar(255,255,0), 3);
      if(!mm_H_gt.empty()) {
        cv::line(mat_for_viewer,
                 cv::Point2f(p[rect_idx[i]].at<float>(0), p[rect_idx[i]].at<float>(1)),
                 cv::Point2f(p[rect_idx[i+1]].at<float>(0), p[rect_idx[i+1]].at<float>(1)),
                 cv::Scalar(0,255,0), 3);
      }  
    }

#if 0
    // This is used for making gif-images
    static int count = 0;
    std::string s_name = std::to_string(count);
    cv::imwrite(s_name + ".png", mat_for_viewer);
    count++;
#endif
    cv::imshow(ms_window_name, mat_for_viewer);
    cv::waitKey(1);

    return;
  }

  void LKTracker::Spin() {
    if(mb_verbose) {
      while(1) {
        if(cv::waitKey(100) == 'q') {
          break;
        }
      }
    }
  }

  void LKTracker::Track(const cv::Mat m_target_image, cv::Mat& m_H, cv::Mat& m_dst_image) {
    // pre-process
    if(mm_H0.empty()) {
      std::cout << "[ERROR] : The Initial Homography hasn't given." << std::endl;
      return;
    }
    
    if(mb_verbose) {
      cv::namedWindow(ms_window_name);
    }

    mm_current_original_image = m_target_image.clone();
    cv::Mat m_current_working_image;
    if(m_target_image.channels() == 3) {
      cv::cvtColor(m_target_image, m_current_working_image, cv::COLOR_BGR2GRAY);
    }
    else {
      m_current_working_image = m_target_image.clone();
    }
    cv::GaussianBlur(m_current_working_image, m_current_working_image, cv::Size(5,5), 0);

    std::vector<cv::Mat> vm_current_image_pyramid 
                         = GenerateImagePyramid(m_current_working_image, m_pyramid_level);

    cv::Mat m_tmp_H = cv::Mat::eye(3,3,CV_32F);
    
    // Coarse-to-Fine Optimization 
    const int use_max_level = m_pyramid_level;
    assert(use_max_level <= m_pyramid_level);

    for (int level = 0; level < use_max_level; level++) {
      m_tmp_H = ProcessInLayer(m_tmp_H,
                               vm_current_image_pyramid[m_pyramid_level - level - 1],
                               mvm_ref_image_pyramid[m_pyramid_level - level - 1],
                               m_pyramid_level - level - 1);
    }

    m_H = m_tmp_H.clone();
    m_dst_image = WarpCurrentImage(mm_current_original_image, m_H);
    if(m_dst_image.type() == 5) {
      m_dst_image.convertTo(m_dst_image, CV_8U, 255);
    }

    return ;
  }

  void LKTracker::UpdateWarp(const cv::Mat& m_params, cv::Mat& H){

  /* PAPER (41) (42) */
  cv::Mat A = cv::Mat::zeros(3,3,CV_32F);
  for(int i = 0; i < 8; i++) {
    A += (m_params.at<float>(i) * mvm_SL3_bases[i]);
  }

  cv::Mat G = cv::Mat::zeros(3, 3, CV_32F);
  cv::Mat A_i = cv::Mat::eye(3, 3, CV_32F);
  float factor_i = 1.f;

  for(int i = 0; i < 9; i++) {
    G += (1.0/factor_i)*(A_i);
    A_i *= A;
    factor_i *= (float)(i+1);
  }

  cv::Mat delta_H = G.clone();

  // update!!
  if(m_type == IC) {
    H = H * delta_H.inv();
  }
  else if(m_type == FC or m_type == ESM) {
    H = H * delta_H;
  }

  return;
}

  cv::Mat LKTracker::WarpCurrentImage(const cv::Mat& src, const cv::Mat& H) {
    cv::Mat dst;
    cv::warpPerspective(src, dst, mm_H0*H,
          cv::Size(m_width, m_height),
          cv::INTER_LINEAR + cv::WARP_INVERSE_MAP, cv::BORDER_CONSTANT,
          cv::Scalar(0,0,0));
    return dst;
  }
}
