#include "YOLOv8.h"
bool Yolov8::ReadModel(cv::dnn::Net& net, std::string& netPath, bool isCuda) {
  try {
    if (!CheckModelPath(netPath)) return false;
    net = cv::dnn::readNetFromONNX(netPath);
#if CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR == 7 && CV_VERSION_REVISION == 0
    net.enableWinograd(
        false);  // bug of opencv4.7.x in AVX only platform
                 // ,https://github.com/opencv/opencv/pull/23112 and
                 // https://github.com/opencv/opencv/issues/23080
                 // net.enableWinograd(true);		//If your CPU supports
                 // AVX2, you
                 // can set it true to speed up
#endif
  } catch (const std::exception&) {
    return false;
  }

  if (isCuda) {
    // cuda
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(
        cv::dnn::DNN_TARGET_CUDA);  // or DNN_TARGET_CUDA_FP16
  } else {
    // cpu
    std::cout << "Inference device: CPU" << std::endl;
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
  }
  return true;
}

// Helper function to draw the predictions on the image
void Yolov8::drawPred(int cow_index, float conf, int left, int top, int right,
                      int bottom, cv::Mat& frame) {
  cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom),
                cv::Scalar(0, 255, 0), 2);

  std::string label = cv::format("%.2f", conf);
  label = "cow " + std::to_string(cow_index) + ": " + label;

  int baseLine;
  cv::Size labelSize =
      cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
  top = std::max(top, labelSize.height);
  cv::rectangle(frame, cv::Point(left, top - labelSize.height),
                cv::Point(left + labelSize.width, top + baseLine),
                cv::Scalar::all(255), cv::FILLED);
  cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.5,
              cv::Scalar());
}

// bool Yolov8::Detect(cv::Mat& srcImg, cv::dnn::Net& net) {
//  // Step 1: Preprocess the image
//  cv::Mat netInputImg;
//  cv::Vec4d params;
//  LetterBox(srcImg, netInputImg, params,
//            cv::Size(input_width, input_height));  //调整输入图片尺寸
//
//  cv::Mat blob;
//  preprocess(netInputImg, blob, input_width, input_height);
//
//  // Step 2: Run forward pass to get output of the output layers
//  net.setInput(blob);
//  std::vector<cv::Mat> outputs;
//  net.forward(outputs, net.getUnconnectedOutLayersNames());
//
//  // Step 3: Postprocess the output to get bounding boxes, confidences, and
//  // class IDs
//
//  std::vector<int> classIds;
//  std::vector<float> confidences;
//  std::vector<cv::Rect> boxes;
//
//  cv::Mat output0 = cv::Mat(cv::Size(outputs[0].size[2], outputs[0].size[1]),
//                            CV_32F, (float*)outputs[0].data)
//                        .t();
//  int rows = output0.rows;
//
//  for (int i = 0; i < rows; ++i) {
//    const float* data = output0.ptr<float>(i);
//    float confidence = data[4];
//    if (confidence >= confidence_threshold) {
//      float* classes_scores = (float*)(data + 4);
//      cv::Mat scores(1, output0.cols - 4, CV_32FC1, classes_scores);
//      cv::Point classIdPoint;
//      double max_class_score;
//      minMaxLoc(scores, 0, &max_class_score, 0, &classIdPoint);
//      if (max_class_score > confidence_threshold &&
//          classIdPoint.x == 0) {  // Assuming class 0 is cow
//        float x = (data[0] - params[2]) / params[0];
//        float y = (data[1] - params[3]) / params[1];
//        float w = data[2] / params[0];
//        float h = data[3] / params[1];
//        int left = MAX(int(x - 0.5 * w + 0.5), 0);
//        int top = MAX(int(y - 0.5 * h + 0.5), 0);
//        classIds.push_back(classIdPoint.x);
//        confidences.push_back(max_class_score);
//        boxes.push_back(cv::Rect(left, top, int(w + 0.5), int(h + 0.5)));
//      }
//    }
//  }
//
//  // Step 4: Apply non-maximum suppression to eliminate redundant overlapping
//  // boxes with lower confidences
//  std::vector<int> indices;
//
//  cv::dnn::NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold,
//                    indices);
//
//  // Step 5: Draw bounding boxes and labels on the image
//  if (indices.empty()) {
//    return false;
//  } else {
//    for (int idx : indices) {
//      cv::Rect box = boxes[idx];
//      drawPred(classIds[idx], confidences[idx], box.x, box.y, box.x +
//      box.width,
//               box.y + box.height, srcImg);
//    }
//    return true;
//  }
//}

DetectResult Yolov8::Detect_with_result(cv::Mat& srcImg, cv::dnn::Net& net) {
  // Step 1: Preprocess the image

  cv::Vec4d params;
  cv::Mat blob;

  preprocess(srcImg, blob, params, input_width, input_height);

  // Step 2: Run forward pass to get output of the output layers
  net.setInput(blob);
  std::vector<cv::Mat> outputs;
  net.forward(outputs, net.getUnconnectedOutLayersNames());

  // Step 3: Postprocess the output to get bounding boxes, confidences, and
  // class IDs

  std::vector<int> classIds;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;
  // outputs.shape = 1 * 1 * 5 * 8400
  cv::Mat output0 = cv::Mat(cv::Size(outputs[0].size[2], outputs[0].size[1]),
      CV_32F, (float*)outputs[0].data).t();
  int rows = output0.rows;

  for (int i = 0; i < rows; ++i) {
    const float* data = output0.ptr<float>(i);
    float confidence = data[4];
    if (confidence >= confidence_threshold) {
      std::cout << "==============confidence=============" << confidence
                << std::endl;
      float* classes_scores = (float*)(data + 4);
      cv::Mat scores(1, output0.cols - 4, CV_32FC1, classes_scores);
      cv::Point classIdPoint;
      double max_class_score;
      minMaxLoc(scores, 0, &max_class_score, 0, &classIdPoint);
      if (max_class_score > confidence_threshold &&
          classIdPoint.x == 0) {  // Assuming class 0 is cow
        float x = (data[0] - params[2]) / params[0];
        float y = (data[1] - params[3]) / params[1];
        float w = data[2] / params[0];
        float h = data[3] / params[1];
        int left = MAX(int(x - 0.5 * w + 0.5), 0);
        int top = MAX(int(y - 0.5 * h + 0.5), 0);
        classIds.push_back(classIdPoint.x);
        confidences.push_back(max_class_score);
        boxes.push_back(cv::Rect(left, top, int(w + 0.5), int(h + 0.5)));
      }
    }
  }

  // Step 4: Apply non-maximum suppression to eliminate redundant overlapping
  // boxes with lower confidences
  std::vector<int> indices;

  cv::dnn::NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold,
                    indices);

  // Step 5: Return
  DetectResult result;
  result.boxes = boxes;
  result.classIds = classIds;
  result.confidences = confidences;
  result.indices = indices;
  return result;
}

bool Yolov8::CheckModelPath(std::string modelPath) {
  if (0 != _access(modelPath.c_str(), 0)) {
    std::cout << "Model path does not exist,  please check " << modelPath
              << std::endl;
    return false;
  } else
    return true;
}

// Helper function to preprocess the input image
void Yolov8::preprocess(const cv::Mat& srcImg, cv::Mat& blob, cv::Vec4d& params,
                        int input_width, int input_height) {
  cv::Mat netInputImg;
  LetterBox(srcImg, netInputImg, params,
            cv::Size(input_width, input_height));  //调整输入图片尺寸

  cv::dnn::blobFromImage(netInputImg, blob, 1.0 / 255.0,
                         cv::Size(input_width, input_height), cv::Scalar(),
                         true, false);
}
void Yolov8::LetterBox(
    const cv::Mat& image, cv::Mat& outImage, cv::Vec4d& params,
    const cv::Size& newShape, bool autoShape, bool scaleFill, bool scaleUp,
    int stride,
    const cv::Scalar& color)  //改变图片的大小，使之符合网络的输入
{
  cv::Size shape = image.size();
  float r = std::min((float)newShape.height / (float)shape.height,
                     (float)newShape.width / (float)shape.width);
  if (!scaleUp) r = std::min(r, 1.0f);


  float ratio[2]{r, r};
  int new_un_pad[2] = {(int)std::round((float)shape.width * r),
                       (int)std::round((float)shape.height * r)};

  auto dw = (float)(newShape.width - new_un_pad[0]);
  auto dh = (float)(newShape.height - new_un_pad[1]);

  if (autoShape) {
    dw = (float)((int)dw % stride);
    dh = (float)((int)dh % stride);
  } else if (scaleFill) {
    dw = 0.0f;
    dh = 0.0f;
    new_un_pad[0] = newShape.width;
    new_un_pad[1] = newShape.height;
    ratio[0] = (float)newShape.width / (float)shape.width;
    ratio[1] = (float)newShape.height / (float)shape.height;
  }

  dw /= 2.0f;
  dh /= 2.0f;

  if (shape.width != new_un_pad[0] && shape.height != new_un_pad[1]) {
    cv::resize(image, outImage, cv::Size(new_un_pad[0], new_un_pad[1]));
  } else {
    outImage = image.clone();
  }

  int top = int(std::round(dh - 0.1f));
  int bottom = int(std::round(dh + 0.1f));
  int left = int(std::round(dw - 0.1f));
  int right = int(std::round(dw + 0.1f));
  params[0] = ratio[0];
  params[1] = ratio[1];
  params[2] = left;
  params[3] = top;
  cv::copyMakeBorder(outImage, outImage, top, bottom, left, right,
                     cv::BORDER_CONSTANT, color);
}
