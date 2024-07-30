#include "DAQ_System.h"

DAQ_System::DAQ_System(QWidget* parent) : QWidget(parent) {
  ui.setupUi(this);
  //========================test opencv
  // tracking=================================
  // List of tracker types in OpenCV 3.4.1
  std::string trackerTypes[1] = {"MIL"};

  // Create a tracker
  std::string trackerType = trackerTypes[0];

#if (CV_MINOR_VERSION < 3)
  { tracker = Tracker::create(trackerType); }
#else
  {
    if (trackerType == "MIL") tracker = cv::TrackerMIL::create();
  }
#endif
  //========================test opencv
  // tracking=================================
  init_cameras();
  timer = new QTimer(this);
  // 加载模型
  std::string model_path = "./weights/best.onnx";
  yolo.ReadModel(net, model_path, true);

  connect(ui.btnStartUp, &QPushButton::clicked, this,
          &DAQ_System::on_startButton_clicked);
  // connect(ui.btnStartUp, &QPushButton::clicked, this,
  //    [=]() {

  //        std::string image_path =
  //        "E:\\Postgraduate\\graduation\\data\\CowDatabase\\RightRgb\\2.png";
  //        //std::string image_path =
  //        "C:\\Users\\Administrator\\Desktop\\1.jpg"; std::string model_path =
  //        "./weigths/best.onnx"; Yolov8 yolo = Yolov8(); cv::dnn::Net net;
  //        yolo.ReadModel(net, model_path, true);
  //        cv::Mat image = cv::imread(image_path);

  //        if (yolo.Detect(image, net)) {
  //            cv::imwrite("C:\\Users\\Administrator\\Desktop\\output.jpg",
  //            image);
  //        }
  //        else {
  //    QMessageBox::critical(nullptr, QString::fromLocal8Bit("错误"),
  //                          QString::fromLocal8Bit("11111！"),
  //                          QMessageBox::Ok);

  //        }
  //    });
  connect(ui.btnCapture, &QPushButton::clicked, this,
          &DAQ_System::on_captureButton_clicked);
  connect(ui.btnShutdown, &QPushButton::clicked, this,
          &DAQ_System::on_stopButton_clicked);
  connect(timer, &QTimer::timeout, this, &DAQ_System::updateFrame);
  connect(this, &DAQ_System::show_img, this, &DAQ_System::showImg);
}

DAQ_System::~DAQ_System() { on_stopButton_clicked(); }
void DAQ_System::closeEvent(QCloseEvent* event) {
  QMessageBox::StandardButton resBtn = QMessageBox::question(
      this, "Confirm Exit", QString::fromLocal8Bit("确定退出程序吗？"),
      QMessageBox::No | QMessageBox::Yes, QMessageBox::Yes);

  if (resBtn != QMessageBox::Yes) {
    event->ignore();
  } else {
    on_stopButton_clicked();
    capturer->close_devices();
    event->accept();
  }
}
void DAQ_System::init_cameras() {
  int color_res = 1;
  int dep_mode = 3;
  int fps = 2;
  std::string root_dir = "F:/DAQ_System/";
  int32_t color_exposure_usec = 33000;
  vector<uint32_t> device_indices{0, 1, 2};
  int32_t powerline_freq = 1;   // default to a 60 Hz powerline，1for 50hz
  float depth_threshold = 2.0;  // default to 2 meter

  int photo_num = 5;  //相机连续拍摄的张数
  xmlparse(device_indices, color_exposure_usec, color_res, dep_mode, fps,
           root_dir, photo_num);
  std::vector<k4a::calibration> cali_list(num_devices);
  // std::string save_dir = create_dir(root_dir);//创建文件夹
  capturer = std::make_unique<MultiDeviceCapturer>(
      device_indices, color_exposure_usec, powerline_freq);
  // MultiDeviceCapturer capturer = MultiDeviceCapturer(device_indices,
  // color_exposure_usec, powerline_freq);
  main_config = get_master_config(color_res, dep_mode, fps);
  secondary_config = get_subordinate_config(color_res, dep_mode, fps);

  for (int i = 0; i < num_devices; i++) {
    if (i == 0) {
      k4a::calibration k4aCalibration =
          capturer->get_master_device().get_calibration(
              main_config.depth_mode, main_config.color_resolution);
      cali_list[i] = k4aCalibration;
    } else {
      k4a::calibration k4aCalibration =
          capturer->get_subordinate_device_by_index(i - 1).get_calibration(
              main_config.depth_mode, main_config.color_resolution);
      cali_list[i] = k4aCalibration;
    }
  }
  xmlset(cali_list);  //设置相关参数
}
void DAQ_System::on_startButton_clicked() {
  if (!isCameraRunning) {
    if (num_devices > k4a::device::get_installed_count()) {
      QMessageBox::critical(nullptr, QString::fromLocal8Bit("错误"),
                            QString::fromLocal8Bit("未连接三个摄像头！"),
                            QMessageBox::Ok);
      return;
    }  //如果设备不够则退出
    capturer->start_devices(main_config, secondary_config);
    isCameraRunning = true;
    timer->start(30);
  }
}

void DAQ_System::on_captureButton_clicked() {
  if (isCameraRunning) {
    vector<k4a::capture> captures =
        capturer->get_synchronized_captures(secondary_config, true);
    std::thread([=]() {
      std::string rootDirPath = "./data";
      auto tp = std::chrono::system_clock::now();
      time_t raw_time = std::chrono::system_clock::to_time_t(tp);
      std::stringstream ss;
      ss << rootDirPath << "/"
         << std::put_time(std::localtime(&raw_time), "%Y-%m-%d-%H-%M-%S");
      std::string save_dir = ss.str();
      if (_mkdir(save_dir.c_str()) == 0) {
        std::vector<std::string> name_list = {"master", "sub1", "sub2"};
        std::string save_dep_path = save_dir + "/depth";
        std::string save_color_path = save_dir + "/color";
        if (_mkdir(save_dep_path.c_str()) == 0 &&
            _mkdir(save_color_path.c_str()) == 0) {
          k4a::image colorImage;
          k4a::image depthImage;
          for (int i = 0; i < 3; i++) {
            k4a::capture tempcapture = captures[i];
            colorImage = tempcapture.get_color_image();
            depthImage = tempcapture.get_depth_image();
            cv::Mat cv_color = color_to_opencv(colorImage);
            cv::Mat cv_depth = depth_to_opencv(depthImage);
            std::string dep_came_path =
                save_dep_path + "/" + name_list[i] + ".png";
            std::string color_came_path =
                save_color_path + "/" + name_list[i] + ".png";
            cv::imwrite(color_came_path, cv_color);
            cv::imwrite(dep_came_path, cv_depth);
            colorImage.reset();
            depthImage.reset();
            cv_color.release();
            cv_depth.release();
          }
        }
      }
    }).detach();
  }
}
void DAQ_System::on_stopButton_clicked() {
  if (isCameraRunning && capturer) {
    std::this_thread::sleep_for(
        std::chrono::milliseconds(2000));  //等待所有存储线程执行完毕
    QImage blackImage(640, 480, QImage::Format_RGB888);
    blackImage.fill(Qt::black);
    ui.mainView->setPixmap(QPixmap::fromImage(blackImage));
    ui.subView1->setPixmap(QPixmap::fromImage(blackImage));
    ui.subView2->setPixmap(QPixmap::fromImage(blackImage));
    capturer->stop_devices();
    isCameraRunning = false;
    timer->stop();
    detected = false;
  }
}
void DAQ_System::updateFrame() {
  if (isCameraRunning && capturer) {
    vector<k4a::capture> captures;
    captures = capturer->get_synchronized_captures(secondary_config, true);
    k4a::image colorImage;
    k4a::image depthImage;
    for (int i = 0; i < 3; i++) {
      k4a::capture tempcapture = captures[i];
      colorImage = tempcapture.get_color_image();
      depthImage = tempcapture.get_depth_image();
      cv::Mat cv_color = color_to_opencv(colorImage);
      cv::Mat cv_depth = depth_to_opencv(depthImage);

      cv::Rect bbox(0, 0, 10, 10);
      cv::Mat color_img_clone = cv_color.clone();

      switch (i) {
        case 0:
          // if (yolo.Detect(color_img_clone, net)) {
          //  emit(show_img(color_img_clone));
          //} else {
          //  emit(show_img(cv_color));
          //}
          // cv::rectangle(cv_color, bbox, cv::Scalar(255, 0, 0), 2, 1);
          count++;
          if (!detected || count % 30 == 0) {
            DetectResult result = yolo.Detect_with_result(color_img_clone, net);
            if (!result.indices.empty()) {
              bbox = result.boxes[result.indices[0]];
              tracker->init(color_img_clone, bbox);
              detected = true;
              count_detect++;
            }
          }
          if (detected) {
            if (tracker->update(color_img_clone, bbox)) {
              count_tracking++;
              std::cout << "===============" << count_detect
                        << "======" << count_tracking << std::endl;
              cv::rectangle(color_img_clone, bbox, cv::Scalar(255, 0, 0), 2, 1);
              emit(show_img(color_img_clone));
              break;
            } else {
              detected = false;
              std::cout << "fail============" << std::endl;
            }
          }
          emit(show_img(color_img_clone));
          break;
        case 1:
          ui.subView1->setPixmap(
              QPixmap::fromImage(opencv_to_QImage(color_img_clone)));
          break;
        case 2:
          ui.subView2->setPixmap(
              QPixmap::fromImage(opencv_to_QImage(color_img_clone)));
          break;
        default:
          break;

          colorImage.reset();
          depthImage.reset();
          color_img_clone.release();
          cv_color.release();
          cv_depth.release();
      }
    }
  }
}
void DAQ_System::showImg(cv::Mat cv_color) {
  ui.mainView->setPixmap(QPixmap::fromImage(opencv_to_QImage(cv_color)));
}
QImage DAQ_System::opencv_to_QImage(cv::Mat cvImg) {
  QImage qImg;
  if (cvImg.channels() == 3)  // 3 channels color image
  {
    cv::cvtColor(cvImg, cvImg, cv::COLOR_BGR2RGB);
    qImg = QImage((const unsigned char*)(cvImg.data), cvImg.cols, cvImg.rows,
                  cvImg.cols * cvImg.channels(), QImage::Format_RGB888);
  } else if (cvImg.channels() == 1)  // grayscale image
  {
    qImg = QImage((const unsigned char*)(cvImg.data), cvImg.cols, cvImg.rows,
                  cvImg.cols * cvImg.channels(), QImage::Format_Indexed8);
  } else {
    qImg = QImage((const unsigned char*)(cvImg.data), cvImg.cols, cvImg.rows,
                  cvImg.cols * cvImg.channels(), QImage::Format_RGB888);
  }
  return qImg;
}
cv::Mat DAQ_System::color_to_opencv(const k4a::image& im) {
  cv::Mat cv_image_with_alpha(im.get_height_pixels(), im.get_width_pixels(),
                              CV_8UC4, (void*)im.get_buffer());
  cv::Mat cv_image_no_alpha;
  cv::cvtColor(cv_image_with_alpha, cv_image_no_alpha, cv::COLOR_BGRA2BGR);
  return cv_image_no_alpha;
}
cv::Mat DAQ_System::depth_to_opencv(const k4a::image& im) {
  return cv::Mat(im.get_height_pixels(), im.get_width_pixels(), CV_16U,
                 (void*)im.get_buffer(),
                 static_cast<size_t>(im.get_stride_bytes()));
}
k4a_device_configuration_t DAQ_System::get_default_config(int color_res,
                                                          int dep_mode,
                                                          int fps)  //共有的配置
{
  k4a_device_configuration_t camera_config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
  camera_config.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;  // rgba
  camera_config.color_resolution =
      (k4a_color_resolution_t)color_res;  // color 分辨率720p=1****************
  camera_config.depth_mode = (k4a_depth_mode_t)
      dep_mode;  // No need for depth during calibration****************
  camera_config.camera_fps =
      (k4a_fps_t)fps;  // Don't use all USB bandwidth***************
  camera_config.subordinate_delay_off_master_usec =
      0;  // Must be zero for master
  camera_config.synchronized_images_only = true;
  return camera_config;
}
k4a_device_configuration_t DAQ_System::get_master_config(int color_res,
                                                         int dep_mode,
                                                         int fps) {
  k4a_device_configuration_t camera_config =
      get_default_config(color_res, dep_mode, fps);
  camera_config.wired_sync_mode = K4A_WIRED_SYNC_MODE_MASTER;  //主机模式
  camera_config.depth_delay_off_color_usec =
      -static_cast<int32_t>(MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC / 2);
  camera_config.synchronized_images_only = true;
  return camera_config;
}
k4a_device_configuration_t DAQ_System::get_subordinate_config(int color_res,
                                                              int dep_mode,
                                                              int fps) {
  k4a_device_configuration_t camera_config =
      get_default_config(color_res, dep_mode, fps);
  camera_config.wired_sync_mode = K4A_WIRED_SYNC_MODE_SUBORDINATE;  //从机模式
  camera_config.depth_delay_off_color_usec =
      MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC / 2;
  return camera_config;
}
void DAQ_System::xmlparse(vector<uint32_t>& device_indices,
                          int32_t& exposure_time, int& color_res, int& dep_mode,
                          int& fps, std::string& root_dir, int& photo_num) {
  const char* xml_path = "./config/config_cpp.xml";
  TiXmlDocument doc;
  doc.LoadFile(xml_path);
  TiXmlElement* arg = doc.FirstChildElement();
  // TiXmlElement* arg = root->FirstChildElement();

  device_indices[0] =
      std::atoi(arg->FirstChildElement("master_idx")->GetText());
  device_indices[1] = std::atoi(arg->FirstChildElement("sub1_idx")->GetText());
  device_indices[2] = std::atoi(arg->FirstChildElement("sub2_idx")->GetText());
  exposure_time = std::atoi(arg->FirstChildElement("exposure")->GetText());
  color_res = std::atoi(arg->FirstChildElement("color_res")->GetText());
  dep_mode = std::atoi(arg->FirstChildElement("dep_mode")->GetText());
  fps = std::atoi(arg->FirstChildElement("fps")->GetText());
  root_dir = arg->FirstChildElement("save_dir")->GetText();
  photo_num = std::atoi(arg->FirstChildElement("photo_num")->GetText());
}
void DAQ_System::xmlset(const vector<k4a::calibration> cali_list) {
  const char* xml_path = "./config/config_cpp.xml";
  TiXmlDocument doc;
  doc.LoadFile(xml_path);
  TiXmlElement* arg = doc.FirstChildElement();
  std::vector<const char*> list_device = {"master", "sub1", "sub2"};
  std::vector<std::string> row_name = {"a", "b", "c", "d"};  //矩阵的行名称
  std::vector<std::string> col_name = {
      "cx", "cy", "fx",   "fy",   "k1", "k2", "k3",           "k4",
      "k5", "k6", "codx", "cody", "p2", "p1", "metric_radius"};
  for (int k = 0; k < 3; k++) {
    const k4a_calibration_extrinsics_t& ex =
        cali_list[k].extrinsics[K4A_CALIBRATION_TYPE_DEPTH]
                               [K4A_CALIBRATION_TYPE_COLOR];  //外参矩阵
    const char* name = list_device[k];
    TiXmlElement* dev = arg->FirstChildElement(name);
    TiXmlElement* mat_d2c = dev->FirstChildElement("d2c");
    for (int i = 1; i <= 4; i++)  //行
    {
      for (int j = 1; j <= 4; j++)  //列
      {
        std::string node_label = row_name[i - 1] + std::to_string(j);
        TiXmlElement* label = mat_d2c->FirstChildElement(node_label.c_str());
        TiXmlNode* temp_node = label->FirstChild();  //需要改变的节点
        if (i > 3) {
          if (j > 3) {
            temp_node->SetValue("1");
          } else {
            temp_node->SetValue("0");
          }
        } else {
          if (j > 3) {
            std::string num_str = std::to_string(ex.translation[i - 1]);
            temp_node->SetValue(num_str.c_str());
          } else {
            std::string num_str =
                std::to_string(ex.rotation[(i - 1) * 3 + (j - 1)]);  //旋转矩阵
            temp_node->SetValue(num_str.c_str());
          }
        }
      }
    }

    TiXmlElement* vec_c2cloud = dev->FirstChildElement("c2cloud");
    TiXmlElement* vec_d2cloud = dev->FirstChildElement("d2cloud");
    for (int s = 0; s < 15; s++) {
      TiXmlElement* node1 = vec_c2cloud->FirstChildElement(col_name[s].c_str());
      TiXmlElement* node2 = vec_d2cloud->FirstChildElement(col_name[s].c_str());
      TiXmlNode* tempnode1 = node1->FirstChild();
      TiXmlNode* tempnode2 = node2->FirstChild();
      std::string node1str = std::to_string(
          cali_list[k].color_camera_calibration.intrinsics.parameters.v[s]);
      tempnode1->SetValue(node1str.c_str());
      std::string node2str = std::to_string(
          cali_list[k].depth_camera_calibration.intrinsics.parameters.v[s]);
      tempnode2->SetValue(node2str.c_str());
    }
  }
  doc.SaveFile(xml_path);
}  //设置xml文件的函数

k4a::image DAQ_System::create_depth_image_like(const k4a::image& im) {
  return k4a::image::create(
      K4A_IMAGE_FORMAT_DEPTH16, im.get_width_pixels(), im.get_height_pixels(),
      im.get_width_pixels() * static_cast<int>(sizeof(uint16_t)));
}
