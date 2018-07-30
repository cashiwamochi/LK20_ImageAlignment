#include <iostream>
#include <opencv2/opencv.hpp>

class LKTracker {
  public:

    enum CalcType {
      FC = 0, // Forward Compositional
      IC = 1, // Inverse Compositional
      ESM = 2, // Efficient Second-Order Mimization
    };

    LKTracker(cv::Mat& image, CalcType t = IC);
    bool Track(const cv::Mat& target_image, cv::Mat& H, cv::Mat& dst_image);
    void SetInitialWarp(cv::Mat& _H0);
    cv::Size GetTrackSize();

  private:
    float ComputeResidual();
    void PreComputeIC();
    void PreComputeFC();
    void PreComputeESM();
    
    void RegisterSL3();
    void WarpCurImg(const cv::Mat &W);
    std::vector<float> ComputeUpdateParams();
    void UpdateWarp(const std::vector<float> vfParams, cv::Mat &W);

    void ComputeHessian(cv::Mat& mHessian, std::vector<cv::Mat>& mvmSteepestDescentImage);

    bool IsConverged(const float& current_error, const float& previous_error);


    /* Reference Image Patch */
    cv::Mat mmRefImage; // for viewer
    cv::Mat mmRefWorkingImage; // this is used for computation

    /* Warped Image Patch */
    cv::Mat mmCurrentImage; // for viewer
    cv::Mat mmCurrentWorkingImage; // this is used for computation
	  cv::Mat mmOriginalWorkingImage;

    cv::Mat mmRefImageDx;
    cv::Mat mmRefImageDy;

    cv::Mat mmW0;

    std::vector<cv::Mat> mvmSL3Bases;
    cv::Mat mmJg;

    int mITER_MAX;
    const int mnHeight, mnWidth;
    const CalcType mType;

    std::vector<cv::Mat> mvmfJw;
    cv::Mat mmHessian;
    std::vector<cv::Mat> mvmSteepestDescentImage;

    std::vector<float> mvfResidual;

};
