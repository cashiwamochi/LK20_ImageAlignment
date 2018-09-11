#include <iostream>
#include <opencv2/opencv.hpp>
#include "Simulator.hpp"
#include "LK20.hpp"


int main(int argc, char* argv[]) {
  if(argc != 2) {
    std::cout << "usage: this.out [path to image]\n";
    return -1;
  }

  // lenna.png
  cv::Mat image = cv::imread(argv[1], 1);

  Simulator _simulator(image);

  //x,y, width, height
  cv::Rect track_rect(210, 210, 160, 160);

  cv::Mat target_template = image(track_rect);
  cv::imwrite("target-template.png", target_template);

  // image, target-rect, pyramid number, how to optimize
  LK20::LKTracker _tracker(image, track_rect, 5, LK20::ESM, LK20::SE3);

  // tx, ty, tz, rx, ry, rz
  cv::Mat warped_image = _simulator.GenerateWarpedImage(-25, -25, 20, 10, -30, -50);

  cv::Mat H0 = (cv::Mat_<float>(3,3) <<  1.f, 0.f, (float)track_rect.x,
                                         0.f, 1.f, (float)track_rect.y,
                                         0.f, 0.f, 1.f);

  // Initial Warp is necessary.
  _tracker.SetInitialWarp(H0);
  // This Ground-Truth is needed in debug-mode (verbose=true)
  _tracker.SetTrueWarp(_simulator.GetWarp());
  _tracker.SetVerbose(true);

  cv::Mat H = cv::Mat::eye(3,3,CV_32F);
  cv::Mat dst_image;
  _tracker.Track(warped_image, H, dst_image);

  cv::imwrite("warped-input.png", dst_image);

  // This keeps debug-window available.
  _tracker.Spin();
  return 0;
}
