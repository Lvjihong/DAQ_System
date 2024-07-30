#pragma once

#include <direct.h>
#include <signal.h>
#include <stdio.h>
#include <windows.h>

#include <chrono>
#include <ctime>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <string>
#include <thread>
#include <vector>

#include "conio.h"
#include "ui_DAQ_System.h"
using std::cerr;
using std::cout;
using std::endl;
using std::vector;
using namespace std;

#include <QCloseEvent>
#include <QMessageBox>
#include <QTimer>
#include <k4a/k4a.hpp>

#include "MultiDeviceCapturer.h"
#include "YOLOv8.h"
#include "tinyxml.h"

class DAQ_System : public QWidget {
  Q_OBJECT

 public:
  DAQ_System(QWidget* parent = nullptr);
  ~DAQ_System();

  void init_cameras();
  void xmlset(const vector<k4a::calibration> cali_list);
  void xmlparse(vector<std::uint32_t>& device_indices, int32_t& exposure_time,
                int& color_res, int& dep_mode, int& fps, std::string& root_dir,
                int& photo_num);
  cv::Mat color_to_opencv(const k4a::image& im);
  cv::Mat depth_to_opencv(const k4a::image& im);
  k4a_device_configuration_t get_master_config(int color_res, int dep_mode,
                                               int fps);
  k4a_device_configuration_t get_subordinate_config(int color_res, int dep_mode,
                                                    int fps);
  // string create_dir(std::string root_dir);
  // bool create_sub_dir(std::string save_path);
  k4a::image create_depth_image_like(const k4a::image& im);
  // void save_dep_color(std::vector<k4a::capture>captures, const std::string
  // save_path);
  k4a_device_configuration_t DAQ_System::get_default_config(int color_res,
                                                            int dep_mode,
                                                            int fps);
  QImage opencv_to_QImage(cv::Mat cvImg);

 protected:
  void closeEvent(QCloseEvent* event) override;
 signals:
  void show_img(cv::Mat);

 private slots:
  void on_startButton_clicked();
  void on_captureButton_clicked();
  void on_stopButton_clicked();
  void updateFrame();
  void showImg(cv::Mat img);

 private:
  Ui::DAQ_SystemClass ui;
  const uint32_t MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC = 160;
  bool isCameraRunning = false;  //是否相机开启
  int cow_indx = 0;              //经过牛的索引

  QTimer* timer;
  std::unique_ptr<MultiDeviceCapturer> capturer;
  k4a_device_configuration_t main_config;
  k4a_device_configuration_t secondary_config;
  size_t num_devices = 3;
  Yolov8 yolo;
  cv::dnn::Net net;

  cv::Ptr<cv::Tracker> tracker;

  int count_detect = 0;
  int count_tracking = 0;
  int count = 0;
  bool detected = false;
};
