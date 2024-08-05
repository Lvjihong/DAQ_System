#include "DAQ_System.h"

DAQ_System::DAQ_System(QWidget* parent) : QWidget(parent) {
  ui.setupUi(this);

  // 创建跟踪器
  tracker = cv::TrackerCSRT::create();
  // 初始化相机，如果没连接摄像头程序无法启动
  init_cameras();
  // 初始化当前记录的目标
  cv::Rect init_bbox = cv::Rect(0, 0, 1, 1);
  current_record_cow.bbox = init_bbox;
  current_record_cow.center_x =
      init_bbox.x + static_cast<float>(init_bbox.width) / 2.0f;
  current_record_cow.center_y =
      init_bbox.y + static_cast<float>(init_bbox.height) / 2.0f;
  current_record_cow.cow_index = cow_index;
  current_record_cow.saved = false;

  timer = new QTimer(this);
  // 加载模型
  std::string model_path = "./weights/last_best.onnx";
  yolo.ReadModel(net, model_path, true);

  connect(ui.btnStartUp, &QPushButton::clicked, this,
          &DAQ_System::on_startButton_clicked);

  // connect(ui.btnCapture, &QPushButton::clicked, this,
  //        &DAQ_System::on_captureButton_clicked);
  connect(ui.btnShutdown, &QPushButton::clicked, this,
          &DAQ_System::on_stopButton_clicked);
  connect(timer, &QTimer::timeout, this, &DAQ_System::updateFrame);
  connect(this, &DAQ_System::show_img, this, &DAQ_System::showImg);
  connect(this, &DAQ_System::save_data, this,
          &DAQ_System::on_captureButton_clicked);
}

DAQ_System::~DAQ_System() { on_stopButton_clicked(); }
// 重写关闭事件
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
// 初始化相机
void DAQ_System::init_cameras() {
  int color_res = 1;
  int dep_mode = 3;
  int fps = 2;
  int32_t color_exposure_usec = 33000;
  vector<uint32_t> device_indices{0, 1, 2};
  int32_t powerline_freq = 1;   // default to a 60 Hz powerline，1for 50hz
  float depth_threshold = 2.0;  // default to 2 meter

  std::string root_dir = "F:/DAQ_System/";
  // 从配置文件获取基本参数
  xmlparse(device_indices, color_exposure_usec, color_res, dep_mode, fps,
           root_dir);
  capturer = std::make_unique<MultiDeviceCapturer>(
      device_indices, color_exposure_usec, powerline_freq);

  // 从相机获取内参
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

  xmlparse(trans_sub2_sub1, trans_main_sub1, cali_list);
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

void DAQ_System::on_captureButton_clicked(const cv::Point& center,
                                          const int cow_index) {
  if (isCameraRunning) {
    vector<k4a::capture> captures =
        capturer->get_synchronized_captures(secondary_config, true);
    std::thread([=]() {
      std::string root_dir_path = "F:/DAQ_System/data/saved_data";
      // 判断目标是否走到中心，true保存两次，false保存一次
      k4a::capture tempcapture = captures[0];
      k4a::image colorImage = tempcapture.get_color_image();
      cv::Mat cv_color = color_to_opencv(colorImage);
      cv::Point view_center(cv_color.cols / 2, cv_color.rows / 2);
      if (cv::norm(view_center - center) < 50 && !current_record_cow.saved) {
        std::string show_dir_path = "F:/DAQ_System/data/show_data/" + std::to_string(current_record_cow.cow_index);
        save_all_data(show_dir_path, true, captures);
        current_record_cow.saved = true;
      }
      auto tp = std::chrono::system_clock::now();
      time_t raw_time = std::chrono::system_clock::to_time_t(tp);
      std::stringstream ss;
      ss << root_dir_path << "/"
         << std::put_time(std::localtime(&raw_time), "%Y-%m-%d-%H-%M-%S");
      std::string save_dir = ss.str();
      save_all_data(save_dir, false, captures);
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

      cv::Rect bbox(0, 0, 1, 1);
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
          if (!detected || count % 15 == 0) {
            DetectResult result = yolo.Detect_with_result(color_img_clone, net);
            if (!result.indices.empty()) {
              bbox = result.boxes[result.indices[0]];
              double iou = calculateIoU(bbox, current_record_cow.bbox);
              if (iou < THRESHOLD_IOU) {
                current_record_cow.cow_index = ++count_detect;
                current_record_cow.saved = false;
                current_record_cow.confidence = result.confidences[0];
              }
              current_record_cow.bbox = bbox;
              current_record_cow.center_x =
                  bbox.x + static_cast<float>(bbox.width) / 2.0f;
              current_record_cow.center_y =
                  bbox.y + static_cast<float>(bbox.height) / 2.0f;

              yolo.drawPred(current_record_cow.cow_index,
                            current_record_cow.confidence, bbox.x, bbox.y,
                            bbox.x + bbox.width, bbox.y + bbox.height,
                            color_img_clone);
              emit(save_data(
                  cv::Point(bbox.x + bbox.width / 2, bbox.y + bbox.height / 2),
                  current_record_cow.cow_index));
              tracker->init(color_img_clone, bbox);
              detected = true;
            } else {
              current_record_cow.bbox = bbox;
              current_record_cow.center_x =
                  bbox.x + static_cast<float>(bbox.width) / 2.0f;
              current_record_cow.center_y =
                  bbox.y + static_cast<float>(bbox.height) / 2.0f;
              detected = false;
              emit(show_img(color_img_clone));
              break;
            }
          }
          if (detected) {
            if (tracker->update(color_img_clone, bbox)) {
              count_tracking++;
              double iou = calculateIoU(bbox, current_record_cow.bbox);
              if (iou < THRESHOLD_IOU) {
                detected = false;
                emit(show_img(color_img_clone));
                break;
              }
              current_record_cow.bbox = bbox;
              current_record_cow.center_x =
                  bbox.x + static_cast<float>(bbox.width) / 2.0f;
              current_record_cow.center_y =
                  bbox.y + static_cast<float>(bbox.height) / 2.0f;
              yolo.drawPred(current_record_cow.cow_index,
                            current_record_cow.confidence, bbox.x, bbox.y,
                            bbox.x + bbox.width, bbox.y + bbox.height,
                            color_img_clone);
              emit(save_data(
                  cv::Point(bbox.x + bbox.width / 2, bbox.y + bbox.height / 2),
                  current_record_cow.cow_index));
              emit(show_img(color_img_clone));
              break;
            } else {
              detected = false;
              std::cout << "===========fail============" << std::endl;
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
      }
      colorImage.reset();
      depthImage.reset();
      color_img_clone.release();
      cv_color.release();
      cv_depth.release();
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
                          int& fps, std::string& root_dir) {
  const char* xml_path = "./config/config_cpp.xml";
  TiXmlDocument doc;
  doc.LoadFile(xml_path);
  TiXmlElement* arg = doc.FirstChildElement();

  std::vector<const char*> list_device = {"master", "sub1", "sub2"};
  std::vector<std::string> row_name = {"a", "b", "c", "d"};  //矩阵的行名称
  std::vector<std::string> col_name = {
      "cx", "cy", "fx",   "fy",   "k1", "k2", "k3",           "k4",
      "k5", "k6", "codx", "cody", "p2", "p1", "metric_radius"};
  TiXmlElement* trans1 = arg->FirstChildElement("trans_sub2_sub1");
  TiXmlElement* trans2 = arg->FirstChildElement("trans_main_sub1");

  device_indices[0] =
      std::atoi(arg->FirstChildElement("master_idx")->GetText());
  device_indices[1] = std::atoi(arg->FirstChildElement("sub1_idx")->GetText());
  device_indices[2] = std::atoi(arg->FirstChildElement("sub2_idx")->GetText());
  exposure_time = std::atoi(arg->FirstChildElement("exposure")->GetText());
  color_res = std::atoi(arg->FirstChildElement("color_res")->GetText());
  dep_mode = std::atoi(arg->FirstChildElement("dep_mode")->GetText());
  fps = std::atoi(arg->FirstChildElement("fps")->GetText());
  root_dir = arg->FirstChildElement("save_dir")->GetText();
}

void DAQ_System::xmlparse(Eigen::Matrix4f& trans_sub2_sub1,
                          Eigen::Matrix4f& trans_main_sub1,
                          std::vector<k4a::calibration>& cali_list) {
  const char* xml_path = "./config/config_cpp.xml";
  TiXmlDocument doc;
  doc.LoadFile(xml_path);
  TiXmlElement* arg = doc.FirstChildElement();
  std::vector<const char*> list_device = {"master", "sub1", "sub2"};
  std::vector<std::string> row_name = {"a", "b", "c", "d"};  //矩阵的行名称
  // std::vector<std::string> col_name = {
  //    "cx", "cy", "fx",   "fy",   "k1", "k2", "k3",           "k4",
  //    "k5", "k6", "codx", "cody", "p2", "p1", "metric_radius" };
  TiXmlElement* trans1 = arg->FirstChildElement("trans_sub2_sub1");
  TiXmlElement* trans2 = arg->FirstChildElement("trans_main_sub1");

  for (int i = 1; i <= 4; i++)  //行
  {
    for (int j = 1; j <= 4; j++)  //列
    {
      std::string name = row_name[i - 1] + std::to_string(j);
      // cout << name;////////////////
      trans_sub2_sub1(i - 1, j - 1) =
          std::atof(trans1->FirstChildElement(name.c_str())->GetText());
      trans_main_sub1(i - 1, j - 1) =
          std::atof(trans2->FirstChildElement(name.c_str())->GetText());
    }
  }
  for (int k = 0; k < 3; k++) {
    k4a_calibration_extrinsics_t& ex =
        cali_list[k]
            .extrinsics[K4A_CALIBRATION_TYPE_DEPTH][K4A_CALIBRATION_TYPE_COLOR];
    const char* name = list_device[k];
    TiXmlElement* dev = arg->FirstChildElement(name);  //设备
    TiXmlElement* mat_d2c = dev->FirstChildElement("d2c");
    for (int i = 1; i <= 4; i++) {
      for (int j = 1; j <= 4; j++) {
        std::string node_label = row_name[i - 1] + std::to_string(j);  //标签名
        if (i > 3) {
          continue;
        } else {
          if (j > 3) {
            ex.translation[i - 1] = std::atof(
                mat_d2c->FirstChildElement(node_label.c_str())->GetText());

          } else {
            ex.rotation[(i - 1) * 3 + (j - 1)] = std::atof(
                mat_d2c->FirstChildElement(node_label.c_str())->GetText());
          }
        }
      }
    }  //外参矩阵
  }
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
k4a::image DAQ_System::create_depth_image_like(int w, int h) {
  return k4a::image::create(K4A_IMAGE_FORMAT_DEPTH16, w, h,
                            w * static_cast<int>(sizeof(uint16_t)));
}
double DAQ_System::calculateIoU(const cv::Rect& rect1, const cv::Rect& rect2) {
  // 计算交集
  cv::Rect intersection = rect1 & rect2;
  int intersectionArea = intersection.area();

  // 如果没有交集，则IoU为0
  if (intersectionArea == 0) return 0.0;

  // 计算并集
  int unionArea = rect1.area() + rect2.area() - intersectionArea;

  // 计算IoU
  double iou = static_cast<double>(intersectionArea) / unionArea;
  return iou;
}
void DAQ_System::save_all_data(const std::string save_dir, const bool need_show,
                               const vector<k4a::capture> captures) {
  if (_mkdir(save_dir.c_str()) == 0) {
    std::vector<std::string> name_list = {"master", "sub1", "sub2"};
    std::string save_dep_path = save_dir + "/depth";
    std::string save_color_path = save_dir + "/color";
    std::string save_point_cloud_path = save_dir + "/point_cloud";
    if (_mkdir(save_dep_path.c_str()) == 0 &&
        _mkdir(save_color_path.c_str()) == 0 &&
        _mkdir(save_point_cloud_path.c_str()) == 0) {
      k4a::image colorImage;
      k4a::image depthImage;
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr result_cloud(
          new pcl::PointCloud<pcl::PointXYZRGB>);
      std::string whole_point_cloud_path =
          save_point_cloud_path + "/" +
          std::to_string(current_record_cow.cow_index) + ".pcd";
      for (int i = 0; i < 3; i++) {
        k4a::capture tempcapture = captures[i];
        colorImage = tempcapture.get_color_image();
        depthImage = tempcapture.get_depth_image();
        cv::Mat cv_color = color_to_opencv(colorImage);
        cv::Mat cv_depth = depth_to_opencv(depthImage);
        std::string dep_came_path = save_dep_path + "/" + name_list[i] + ".png";
        std::string color_came_path =
            save_color_path + "/" + name_list[i] + ".png";
        // std::string point_cloud_path =
        //    save_point_cloud_path + "/" + name_list[i] + ".pcd";
        // 保存rgb和depth图像
        cv::imwrite(color_came_path, cv_color);
        cv::imwrite(dep_came_path, cv_depth);
        if (need_show) {
          // 生成点云
          k4a::image dep_data = k4a::image::create_from_buffer(
              K4A_IMAGE_FORMAT_DEPTH16, cv_depth.size().width,
              cv_depth.size().height,
              cv_depth.size().width * static_cast<int>(sizeof(uint16_t)),
              cv_depth.data,
              cv_depth.size().height * cv_depth.size().width *
                  static_cast<int>(sizeof(uint8_t)),
              NULL, NULL);
          k4a::transformation trans(cali_list[i]);

          k4a::image trans_dep = create_depth_image_like(
              cv_color.size().width, cv_color.size().height);
          k4a::image cloud_image = k4a::image::create(
              K4A_IMAGE_FORMAT_CUSTOM, cv_color.size().width,
              cv_color.size().height,
              cv_color.size().width * 3 * (int)sizeof(int16_t));  //点云
          trans.depth_image_to_color_camera(dep_data, &trans_dep);

          trans.depth_image_to_point_cloud(
              trans_dep, K4A_CALIBRATION_TYPE_COLOR, &cloud_image);
          pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(
              new pcl::PointCloud<pcl::PointXYZRGB>);
          cloud->is_dense = true;
          const int16_t* cloud_image_data =
              reinterpret_cast<const int16_t*>(cloud_image.get_buffer());
          int wid = cv_color.size().width;  //图片宽度
          for (size_t h = 0; h < cv_color.size().height; h++) {
            for (size_t w = 0; w < cv_color.size().width; w++) {
              pcl::PointXYZRGB point;
              size_t indx0 = h * wid + w;
              point.x = cloud_image_data[3 * indx0 + 0] / 1000.0f;
              point.y = cloud_image_data[3 * indx0 + 1] / 1000.0f;
              point.z = cloud_image_data[3 * indx0 + 2] / 1000.0f;

              point.b = cv_color.at<cv::Vec3b>(h, w)[0];
              point.g = cv_color.at<cv::Vec3b>(h, w)[1];
              point.r = cv_color.at<cv::Vec3b>(h, w)[2];
              if (point.x == 0 && point.y == 0 && point.z == 0) continue;
              cloud->push_back(point);
            }
          }
          // pcl::io::savePCDFileASCII(point_cloud_path, *cloud);
          if (i == 0) {
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr trans_cloud(
                new pcl::PointCloud<pcl::PointXYZRGB>);
            pcl::transformPointCloud(*cloud, *trans_cloud,
                                     trans_main_sub1);  // cow inverse sheep no
            *result_cloud = (*result_cloud) + (*trans_cloud);

          } else if (i == 2) {
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr trans_cloud(
                new pcl::PointCloud<pcl::PointXYZRGB>);
            pcl::transformPointCloud(*cloud, *trans_cloud, trans_sub2_sub1);
            *result_cloud = (*result_cloud) + (*trans_cloud);
          } else {
            *result_cloud = (*result_cloud) + (*cloud);
          }
        }
        colorImage.reset();
        depthImage.reset();
        cv_color.release();
        cv_depth.release();
      }
      if (need_show) {
        pcl::io::savePCDFileASCII(whole_point_cloud_path, *result_cloud);
      }
    }
  }
}
