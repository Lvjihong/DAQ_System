#pragma once
#include <io.h>

#include <opencv2/opencv.hpp>
struct DetectResult {
  std::vector<int> classIds;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;
  std::vector<int> indices;
};
class Yolov8 {
 public:
  Yolov8() {}
  ~Yolov8() {}

  bool ReadModel(cv::dnn::Net& net, std::string& netPath, bool isCuda);
  bool Detect(cv::Mat& srcImg, cv::dnn::Net& net);
  DetectResult Detect_with_result(cv::Mat& srcImg,
                                             cv::dnn::Net& net);
  void preprocess(const cv::Mat& image, cv::Mat& blob, int input_width,
                  int input_height);
  void drawPred(int classId, float conf, int left, int top, int right,
                int bottom, cv::Mat& frame);
  bool CheckModelPath(std::string modelPath);
  void LetterBox(const cv::Mat& image, cv::Mat& outImage,
                 cv::Vec4d& params,  //[ratio_x,ratio_y,dw,dh]
                 const cv::Size& newShape = cv::Size(640, 640),
                 bool autoShape = false, bool scaleFill = false,
                 bool scaleUp = true, int stride = 32,
                 const cv::Scalar& color = cv::Scalar(
                     114, 114, 114));  //改变图片的大小，使之符合网络的输入

 private:
  std::vector<std::string> _className = {"cow"};
  const int input_width = 640;
  const int input_height = 640;
  const float confidence_threshold =
      0.05;  // Minimum confidence for a detection to be considered
  const float nms_threshold = 0.5;  // Non-maximum suppression threshold
};