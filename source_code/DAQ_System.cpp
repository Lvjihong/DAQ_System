#include "DAQ_System.h"

DAQ_System::DAQ_System(QWidget* parent) : QWidget(parent) {
	ui.setupUi(this);

	// 检查 OpenCV 是否启用了 CUDA 支持
	bool hasCuda = cv::cuda::getCudaEnabledDeviceCount() > 0;

	if (hasCuda) {
		std::cout << "OpenCV is built with CUDA support." << std::endl;
	}
	else {
		std::cout << "OpenCV is not built with CUDA support." << std::endl;
	}

	// 创建跟踪器
	tracker = cv::TrackerCSRT::create();
	// 初始化相机，如果没连接摄像头程序无法启动
	init_cameras();
	// 初始化当前记录的目标
	cv::Rect init_bbox = cv::Rect(1280, 0, 1, 1);
	current_record_cow.bbox = init_bbox;
	current_record_cow.center_x =
		init_bbox.x + static_cast<float>(init_bbox.width) / 2.0f;
	current_record_cow.center_y =
		init_bbox.y + static_cast<float>(init_bbox.height) / 2.0f;
	current_record_cow.cow_index = cow_index;
	current_record_cow.saved = false;
	current_record_cow.confidence = -1;

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
	connect(this, &DAQ_System::img_ready, this, &DAQ_System::showImg);
	connect(this, &DAQ_System::save_data, this,
		&DAQ_System::on_captureButton_clicked);

	connect(ui.btnStartRecord, &QPushButton::clicked, [=] {
		if (!isCameraRunning) {
			if (num_devices > k4a::device::get_installed_count()) {
				QMessageBox::critical(nullptr, QString::fromLocal8Bit("错误"),
					QString::fromLocal8Bit("未连接摄像头！"),
					QMessageBox::Ok);
				return;
			}  //如果设备不够则退出
			device.start_cameras(&config);
			int iAuto = 0;       //用来稳定，类似自动曝光
			int iAutoError = 0;  // 统计自动曝光的失败次数
			while (true) {
				if (device.get_capture(&capture)) {
					std::cout << iAuto << ". Capture several frames to give auto-exposure"
						<< std::endl;

					//跳过前 n 个（成功的数据采集）循环，用来稳定
					if (iAuto != 30) {
						iAuto++;
						continue;
					}
					else {
						std::cout << "Done: auto-exposure" << std::endl;
						break;  // 跳出该循环，完成相机的稳定过程
					}

				}
				else {
					std::cout << iAutoError << ". K4A_WAIT_RESULT_TIMEOUT." << std::endl;
					if (iAutoError != 30) {
						iAutoError++;
						continue;
					}
					else {
						std::cout << "Error: failed to give auto-exposure. " << std::endl;
						return;
					}
				}
			}
			isCameraRunning = true;
			startThreads();
		}
		});

}

DAQ_System::~DAQ_System() {
	on_stopButton_clicked();

}
// 重写关闭事件
void DAQ_System::closeEvent(QCloseEvent* event) {
	QMessageBox::StandardButton resBtn = QMessageBox::question(
		this, "Confirm Exit", QString::fromLocal8Bit("确定退出程序吗？"),
		QMessageBox::No | QMessageBox::Yes, QMessageBox::Yes);

	if (resBtn != QMessageBox::Yes) {
		event->ignore();
	}
	else {
		on_stopButton_clicked();
		device.close();
		event->accept();
	}
}
// 初始化相机
void DAQ_System::init_cameras() {
	int color_res = 1;
	int dep_mode = 3;
	int fps = 1;
	int32_t color_exposure_usec = 33000;
	int32_t powerline_freq = 1;  // default to a 60 Hz powerline，1for 50hz
	xml_parse(color_res, dep_mode, color_exposure_usec, fps, root_dir_path, THRESHOLD_DISTANCE, THRESHOLD_BBOX_WIDTH, THRESHOLD_BBOX_HEIGHT);
	const uint32_t device_count = k4a::device::get_installed_count();
	if (0 == device_count) {
		cout << "Error: no K4A devices found. " << endl;
		return;
	}
	else {
		std::cout << "Found " << device_count << " connected devices. "
			<< std::endl;
		if (1 != device_count)  // 超过1个设备，也输出错误信息。
		{
			std::cout << "Error: more than one K4A devices found. " << std::endl;
			return;
		}
		else  // 该示例代码仅限对1个设备操作
		{
			std::cout << "Done: found 1 K4A device. " << std::endl;
		}
	}
	//打开（默认）设备
	device = k4a::device::open(K4A_DEVICE_DEFAULT);
	device.set_color_control(K4A_COLOR_CONTROL_EXPOSURE_TIME_ABSOLUTE,
		K4A_COLOR_CONTROL_MODE_MANUAL, color_exposure_usec);
	device.set_color_control(K4A_COLOR_CONTROL_POWERLINE_FREQUENCY,
		K4A_COLOR_CONTROL_MODE_MANUAL, powerline_freq);
	config = get_default_config(color_res, dep_mode, fps);
	config.synchronized_images_only = true;
	k4a::calibration k4aCalibration =
		device.get_calibration(config.depth_mode, config.color_resolution);
	cali = k4aCalibration;
	std::cout << "Done: open device. " << std::endl;
	xml_set(cali);
}
void DAQ_System::on_startButton_clicked() {
	if (!isCameraRunning) {
		if (num_devices > k4a::device::get_installed_count()) {
			QMessageBox::critical(nullptr, QString::fromLocal8Bit("错误"),
				QString::fromLocal8Bit("未连接摄像头！"),
				QMessageBox::Ok);
			return;
		}  //如果设备不够则退出
		device.start_cameras(&config);
		int iAuto = 0;       //用来稳定，类似自动曝光
		int iAutoError = 0;  // 统计自动曝光的失败次数
		while (true) {
			if (device.get_capture(&capture)) {
				std::cout << iAuto << ". Capture several frames to give auto-exposure"
					<< std::endl;

				//跳过前 n 个（成功的数据采集）循环，用来稳定
				if (iAuto != 30) {
					iAuto++;
					continue;
				}
				else {
					std::cout << "Done: auto-exposure" << std::endl;
					break;  // 跳出该循环，完成相机的稳定过程
				}

			}
			else {
				std::cout << iAutoError << ". K4A_WAIT_RESULT_TIMEOUT." << std::endl;
				if (iAutoError != 30) {
					iAutoError++;
					continue;
				}
				else {
					std::cout << "Error: failed to give auto-exposure. " << std::endl;
					return;
				}
			}
		}
		isCameraRunning = true;
		timer->setInterval(60);
		timer->start(30);
	}
}

void DAQ_System::on_captureButton_clicked(const cv::Point& center,
	const int cow_index,
	const k4a::capture capture) {
	if (isCameraRunning) {
		std::thread([=]() {
			std::string root_dir_path = "./data/saved_data";
			// 判断目标是否走到中心，true保存两次，false保存一次
			k4a::image colorImage = capture.get_color_image();
			cv::Mat cv_color = color_to_opencv(colorImage);
			cv::Point view_center(cv_color.cols / 2, cv_color.rows / 2);
			if (cv::norm(view_center - center) < 100 && !current_record_cow.saved) {
				std::string show_dir_path =
					"F:/DAQ_System/data/show_data/" +
					std::to_string(current_record_cow.cow_index);
				save_all_data(show_dir_path, true, capture);
				current_record_cow.saved = true;
			}
			auto tp = std::chrono::system_clock::now();
			time_t raw_time = std::chrono::system_clock::to_time_t(tp);
			std::stringstream ss;
			ss << root_dir_path << "/"
				<< std::put_time(std::localtime(&raw_time), "%Y-%m-%d-%H-%M-%S");
			std::string save_dir =
				ss.str() + "cow" + std::to_string(current_record_cow.cow_index);
			save_all_data(save_dir, false, capture);
			}).detach();
	}
}
void DAQ_System::on_stopButton_clicked() {
	if (isCameraRunning && capture) {
		capture.reset();
		device.stop_cameras();
		isCameraRunning = false;
		timer->stop();
		need_detect = true;
	}
	while (true) {
		if (rgb_queue.empty() && depth_queue.empty() && img_with_bbox_queue.empty()) {
			std::this_thread::sleep_for(
				std::chrono::milliseconds(2000));  //等待所有存储线程执行完毕
			QImage blackImage(640, 480, QImage::Format_RGB888);
			blackImage.fill(Qt::black);
			ui.mainView->setPixmap(QPixmap::fromImage(blackImage));
			ui.subView1->setPixmap(QPixmap::fromImage(blackImage));
			break;
		}
	}
}
void DAQ_System::updateFrame() {
	device.get_capture(&capture);
	if (isCameraRunning && capture) {
		k4a::image colorImage;
		k4a::image depthImage;
		// 获取相机（只有单视角）
		colorImage = capture.get_color_image();
		depthImage = capture.get_depth_image();
		cv::Mat cv_color = color_to_opencv(colorImage);
		cv::Mat cv_depth = depth_to_opencv(depthImage);

		cv::Rect bbox(0, 0, 1, 1);
		cv::Mat color_img_clone = cv_color.clone();

		count++;
		if (need_detect || count % 5 == 0) {
			DetectResult result = yolo.Detect_with_result(color_img_clone, net);
			if (!result.indices.empty()) {
				bbox = result.boxes[result.indices[0]];
				double iou = calculateIoU(bbox, current_record_cow.bbox);
				if (iou < THRESHOLD_IOU) {
					current_record_cow.cow_index = ++count_detect;
					current_record_cow.saved = false;
					current_record_cow.confidence = result.confidences[0];
				}
				yolo.drawPred(
					current_record_cow.cow_index, current_record_cow.confidence, bbox.x,
					bbox.y, bbox.x + bbox.width, bbox.y + bbox.height, color_img_clone);
				emit(save_data(
					cv::Point(bbox.x + bbox.width / 2, bbox.y + bbox.height / 2),
					current_record_cow.cow_index, capture));
				tracker->init(color_img_clone, bbox);
				need_detect = false;
			}
			else {
				need_detect = true;
			}
			current_record_cow.bbox = bbox;
			current_record_cow.center_x =
				bbox.x + static_cast<float>(bbox.width) / 2.0f;
			current_record_cow.center_y =
				bbox.y + static_cast<float>(bbox.height) / 2.0f;
		}
		if (!need_detect) {
			if (tracker->update(color_img_clone, bbox)) {
				double iou = calculateIoU(bbox, current_record_cow.bbox);
				if (iou < THRESHOLD_IOU) {
					need_detect = true;
					emit(img_ready(color_img_clone, cv_depth));
					current_record_cow.bbox = bbox;
					current_record_cow.center_x =
						bbox.x + static_cast<float>(bbox.width) / 2.0f;
					current_record_cow.center_y =
						bbox.y + static_cast<float>(bbox.height) / 2.0f;
					return;
				}
				current_record_cow.bbox = bbox;
				current_record_cow.center_x =
					bbox.x + static_cast<float>(bbox.width) / 2.0f;
				current_record_cow.center_y =
					bbox.y + static_cast<float>(bbox.height) / 2.0f;
				yolo.drawPred(
					current_record_cow.cow_index, current_record_cow.confidence, bbox.x,
					bbox.y, bbox.x + bbox.width, bbox.y + bbox.height, color_img_clone);
				emit(save_data(
					cv::Point(bbox.x + bbox.width / 2, bbox.y + bbox.height / 2),
					current_record_cow.cow_index, capture));
			}
			else {
				need_detect = true;
				std::cout << "===========fail============" << std::endl;
			}
		}
		emit(img_ready(color_img_clone, cv_depth));

		colorImage.reset();
		depthImage.reset();
		color_img_clone.release();
		cv_color.release();
		cv_depth.release();
	}
}
void DAQ_System::showImg(cv::Mat img_rgb, cv::Mat img_depth) {
	ui.mainView->setPixmap(QPixmap::fromImage(opencv_to_QImage(img_rgb)));
	ui.mainView->setAlignment(Qt::AlignCenter);
	// 将深度图归一化到 [0, 255] 范围，便于映射为彩色图
	cv::Mat normalized_depth;
	cv::normalize(img_depth, normalized_depth, 0, 255, cv::NORM_MINMAX, CV_8U);

	// 使用 applyColorMap 函数将其转换为伪彩色图像
	cv::Mat pseudo_color_image;
	cv::applyColorMap(normalized_depth, pseudo_color_image, cv::COLORMAP_JET);
	ui.subView1->setPixmap(QPixmap::fromImage(opencv_to_QImage(pseudo_color_image)));
	ui.subView1->setAlignment(Qt::AlignCenter);
}

QImage DAQ_System::opencv_to_QImage(cv::Mat cvImg) {
	QImage qImg;
	if (cvImg.channels() == 3)  // 3 channels color image
	{
		cv::cvtColor(cvImg, cvImg, cv::COLOR_BGR2RGB);
		qImg = QImage((const unsigned char*)(cvImg.data), cvImg.cols, cvImg.rows,
			cvImg.cols * cvImg.channels(), QImage::Format_RGB888);
	}
	else if (cvImg.channels() == 1)  // grayscale image
	{
		qImg = QImage((const unsigned char*)(cvImg.data), cvImg.cols, cvImg.rows,
			cvImg.cols * cvImg.channels(), QImage::Format_Indexed8);
	}
	else {
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

	std::vector<const char*> list_device = { "master", "sub1", "sub2" };
	std::vector<std::string> row_name = { "a", "b", "c", "d" };  //矩阵的行名称
	std::vector<std::string> col_name = {
		"cx", "cy", "fx",   "fy",   "k1", "k2", "k3",           "k4",
		"k5", "k6", "codx", "cody", "p2", "p1", "metric_radius" };
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

void DAQ_System::xml_parse(int& color_res, int& dep_mode, int& exposure, int& fps, std::string& save_dir, double& threshold_distance, int& threshold_bbox_width, int& threshold_bbox_height)
{
	std::string filePath = "./config/config.xml";

	// 加载XML文件
	TiXmlDocument doc(filePath.c_str());
	if (!doc.LoadFile()) {
		std::cerr << "Failed to load XML file: " << doc.ErrorDesc() << std::endl;
		return;
	}

	// 获取根节点
	TiXmlElement* root = doc.FirstChildElement("Argument");
	if (!root) {
		std::cerr << "Failed to find <Argument> element in XML." << std::endl;
		return;
	}

	// 解析color_res
	TiXmlElement* element = root->FirstChildElement("color_res");
	if (element) {
		color_res = std::stoi(element->GetText());
	}

	// 解析dep_mode
	element = root->FirstChildElement("dep_mode");
	if (element) {
		dep_mode = std::stoi(element->GetText());
	}

	// 解析exposure
	element = root->FirstChildElement("exposure");
	if (element) {
		exposure = std::stoi(element->GetText());
	}

	// 解析fps
	element = root->FirstChildElement("fps");
	if (element) {
		fps = std::stoi(element->GetText());
	}

	// 解析save_dir
	element = root->FirstChildElement("save_dir");
	if (element) {
		save_dir = element->GetText();
	}

	// 解析threshold_distance
	element = root->FirstChildElement("threshold_distance");
	if (element) {
		threshold_distance = std::stoi(element->GetText());
	}

	// 解析threshold_bbox_width
	element = root->FirstChildElement("threshold_bbox_width");
	if (element) {
		threshold_bbox_width = std::stoi(element->GetText());
	}

	// 解析threshold_bbox_height
	element = root->FirstChildElement("threshold_bbox_height");
	if (element) {
		threshold_bbox_height = std::stoi(element->GetText());
	}
}


void DAQ_System::xml_set(k4a::calibration cali) {
	const char* xml_path = "./config/config_cpp.xml";
	TiXmlDocument doc;
	doc.LoadFile(xml_path);
	TiXmlElement* arg = doc.FirstChildElement();
	std::vector<std::string> row_name = { "a", "b", "c", "d" };  //矩阵的行名称
	std::vector<std::string> col_name = {
		"cx", "cy", "fx",   "fy",   "k1", "k2", "k3",           "k4",
		"k5", "k6", "codx", "cody", "p2", "p1", "metric_radius" };
	const k4a_calibration_extrinsics_t& ex =
		cali.extrinsics[K4A_CALIBRATION_TYPE_DEPTH]
		[K4A_CALIBRATION_TYPE_COLOR];  //外参矩阵
	TiXmlElement* dev = arg->FirstChildElement("cali");
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
				}
				else {
					temp_node->SetValue("0");
				}
			}
			else {
				if (j > 3) {
					std::string num_str = std::to_string(ex.translation[i - 1]);
					temp_node->SetValue(num_str.c_str());
				}
				else {
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
			cali.color_camera_calibration.intrinsics.parameters.v[s]);
		tempnode1->SetValue(node1str.c_str());
		std::string node2str = std::to_string(
			cali.depth_camera_calibration.intrinsics.parameters.v[s]);
		tempnode2->SetValue(node2str.c_str());
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
double DAQ_System::calculate_distance_of_centers_bbox(const cv::Rect& rect1, const cv::Rect& rect2) {
	// 计算第一个矩形的中心点的横坐标
	double centerX1 = rect1.x + rect1.width / 2.0;

	// 计算第二个矩形的中心点的横坐标
	double centerX2 = rect2.x + rect2.width / 2.0;

	// 计算横坐标的距离
	double horizontal_distance = std::abs(centerX2 - centerX1);

	return horizontal_distance;
}
void DAQ_System::save_all_data(const std::string save_dir, const bool need_show,
	const k4a::capture capture) {
	if (_mkdir(save_dir.c_str()) == 0) {
		std::string save_dep_path = save_dir + "/depth";
		std::string save_color_path = save_dir + "/color";
		std::string save_point_cloud_path = save_dir + "/point_cloud";
		bool flag = false;
		if (need_show) {
			flag = _mkdir(save_point_cloud_path.c_str());
		}
		if (_mkdir(save_dep_path.c_str()) == 0 &&
			_mkdir(save_color_path.c_str()) == 0 && !flag) {
			k4a::image colorImage;
			k4a::image depthImage;
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr result_cloud(
				new pcl::PointCloud<pcl::PointXYZRGB>);

			colorImage = capture.get_color_image();
			depthImage = capture.get_depth_image();
			cv::Mat cv_color = color_to_opencv(colorImage);
			cv::Mat cv_depth = depth_to_opencv(depthImage);
			// 获取当前时间点
			auto now = std::chrono::system_clock::now();

			// 获取自纪元（1970-01-01）以来的毫秒数
			auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

			std::string dep_came_path = save_dep_path + "/" + std::to_string(timestamp) + ".png";
			std::string color_came_path =
				save_color_path + "/" + std::to_string(timestamp) + ".png";
			std::string point_cloud_path =
				save_point_cloud_path + "/" + std::to_string(timestamp) + ".pcd";
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
				k4a::transformation trans(cali);

				k4a::image trans_dep = create_depth_image_like(cv_color.size().width,
					cv_color.size().height);
				k4a::image cloud_image = k4a::image::create(
					K4A_IMAGE_FORMAT_CUSTOM, cv_color.size().width,
					cv_color.size().height,
					cv_color.size().width * 3 * (int)sizeof(int16_t));  //点云
				trans.depth_image_to_color_camera(dep_data, &trans_dep);

				trans.depth_image_to_point_cloud(trans_dep, K4A_CALIBRATION_TYPE_COLOR,
					&cloud_image);
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
				pcl::io::savePCDFileASCII(point_cloud_path, *cloud);
			}
			colorImage.reset();
			depthImage.reset();
			cv_color.release();
			cv_depth.release();
		}
	}
}

void DAQ_System::save_all_data(const std::string save_path, const bool need_show, const cv::Mat color_img, const cv::Mat depth_img)
{
	std::string color_path = save_path + "_color.png";
	std::string dep_path = save_path + "_depth.png";
	std::string point_cloud_path = save_path + "_point_cloud.pcd";
	// 保存rgb和depth图像
	cv::imwrite(color_path, color_img);
	cv::imwrite(dep_path, depth_img);

	if (need_show) {
		// 生成点云
		k4a::image dep_data = k4a::image::create_from_buffer(
			K4A_IMAGE_FORMAT_DEPTH16, depth_img.size().width,
			depth_img.size().height,
			depth_img.size().width * static_cast<int>(sizeof(uint16_t)),
			depth_img.data,
			depth_img.size().height * depth_img.size().width *
			static_cast<int>(sizeof(uint8_t)),
			NULL, NULL);
		k4a::transformation trans(cali);

		k4a::image trans_dep = create_depth_image_like(color_img.size().width,
			color_img.size().height);
		k4a::image cloud_image = k4a::image::create(
			K4A_IMAGE_FORMAT_CUSTOM, color_img.size().width,
			color_img.size().height,
			color_img.size().width * 3 * (int)sizeof(int16_t));  //点云
		trans.depth_image_to_color_camera(dep_data, &trans_dep);

		trans.depth_image_to_point_cloud(trans_dep, K4A_CALIBRATION_TYPE_COLOR,
			&cloud_image);
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(
			new pcl::PointCloud<pcl::PointXYZRGB>);
		cloud->is_dense = true;
		const int16_t* cloud_image_data =
			reinterpret_cast<const int16_t*>(cloud_image.get_buffer());
		int wid = color_img.size().width;  //图片宽度
		for (size_t h = 0; h < color_img.size().height; h++) {
			for (size_t w = 0; w < color_img.size().width; w++) {
				pcl::PointXYZRGB point;
				size_t indx0 = h * wid + w;
				point.x = cloud_image_data[3 * indx0 + 0] / 1000.0f;
				point.y = cloud_image_data[3 * indx0 + 1] / 1000.0f;
				point.z = cloud_image_data[3 * indx0 + 2] / 1000.0f;

				point.b = color_img.at<cv::Vec3b>(h, w)[0];
				point.g = color_img.at<cv::Vec3b>(h, w)[1];
				point.r = color_img.at<cv::Vec3b>(h, w)[2];
				if (point.x == 0 && point.y == 0 && point.z == 0) continue;
				cloud->push_back(point);
			}
		}
		pcl::io::savePCDFileASCII(point_cloud_path, *cloud);
	}
}


void DAQ_System::startThreads() {
	capture_thread = std::thread(&DAQ_System::image_capture_thread, this);
	handle_thread = std::thread(&DAQ_System::image_handle_thread, this);
	save_thread = std::thread(&DAQ_System::image_save_thread, this);

	capture_thread.detach();
	handle_thread.detach();
	save_thread.detach();
}

void DAQ_System::image_capture_thread() {
	while (isCameraRunning && capture) {
		if (device.get_capture(&capture)) {
			k4a::image rgb_image = capture.get_color_image();
			cv::Mat rgb_mat = color_to_opencv(rgb_image);

			k4a::image depth_image = capture.get_depth_image();
			cv::Mat depth_mat = depth_to_opencv(depth_image);

			{
				std::lock_guard<std::mutex> locker(rgb_mutex);
				rgb_queue.push(rgb_mat.clone());
			}
			rgb_condition.notify_one();

			{
				std::lock_guard<std::mutex> locker(depth_mutex);
				depth_queue.push(depth_mat.clone());
			}
			depth_condition.notify_one();

			std::this_thread::sleep_for(std::chrono::milliseconds(200));

			rgb_image.reset();
			rgb_mat.release();
			depth_image.reset();
			depth_mat.release();
		}
		else {
			break;
		}
	}
}

void DAQ_System::image_handle_thread() {
	while (true) {
		cv::Mat rgb_frame;
		{
			std::unique_lock<std::mutex> locker(rgb_mutex);
			rgb_condition.wait(locker, [this] { return !rgb_queue.empty(); });
			rgb_frame = rgb_queue.front();
			rgb_queue.pop();
		}

		cv::Rect bbox(0, 0, 1, 1);
		cv::Mat color_img_clone = rgb_frame.clone();

		count++;
		float confidence = 0.0;
		bool is_init = false;
		if (need_detect || count % 5 == 0) {
			DetectResult result = yolo.Detect_with_result(color_img_clone, net);
			if (!result.indices.empty()) {
				bbox = result.boxes[result.indices[0]];
				tracker->init(color_img_clone, bbox);
				is_init = true;
				confidence = result.confidences[0];
				need_detect = false;
			}
			else {
				need_detect = true;
			}
		}
		if (!is_init) {
			if (!need_detect) {
				if (tracker->update(color_img_clone, bbox)) {
					std::cout << "===========track success!============" << std::endl;
					confidence = current_record_cow.confidence;
				}
				else
				{
					need_detect = true;
					std::cout << "===========track failing!============" << std::endl;
				}
			}
		}

		color_img_with_bbox img_with_bbox;
		img_with_bbox.bbox = bbox;
		img_with_bbox.img = color_img_clone;
		img_with_bbox.confidence = confidence;

		{
			std::lock_guard<std::mutex> locker(bbox_mutex);
			img_with_bbox_queue.push(img_with_bbox);
		}
		bbox_condition.notify_one();

		color_img_clone.release();
		rgb_frame.release();
	}
}

void DAQ_System::image_save_thread() {
	while (true) {
		cv::Mat depth_frame;
		{
			std::unique_lock<std::mutex> locker(depth_mutex);
			depth_condition.wait(locker, [this] { return !depth_queue.empty(); });
			depth_frame = depth_queue.front();
			depth_queue.pop();
		}
		color_img_with_bbox img_object;
		{
			std::unique_lock<std::mutex> locker(bbox_mutex);
			bbox_condition.wait(locker, [this] { return !img_with_bbox_queue.empty(); });
			img_object = img_with_bbox_queue.front();
			img_with_bbox_queue.pop();
		}
		// bbox太小，不显示bbox也不保存
		if (img_object.bbox.width < 600 || img_object.bbox.height < 350) {
			emit(img_ready(img_object.img, depth_frame));
			continue;
		}

		// id管理
		//double iou = calculateIoU(img_object.bbox, current_record_cow.bbox);
		int distance = calculate_distance_of_centers_bbox(img_object.bbox, current_record_cow.bbox);
		if (distance > THRESHOLD_DISTANCE)
		{
			current_record_cow.cow_index++;
			current_record_cow.saved = false;
			current_record_cow.confidence = img_object.confidence;
		}
		current_record_cow.bbox = img_object.bbox;
		current_record_cow.center_x =
			img_object.bbox.x + static_cast<float>(img_object.bbox.width) / 2.0f;
		current_record_cow.center_y =
			img_object.bbox.y + static_cast<float>(img_object.bbox.height) / 2.0f;
		// 显示图像
		cv::Mat color_img_clone = img_object.img.clone();
		yolo.drawPred(
			current_record_cow.cow_index, current_record_cow.confidence, img_object.bbox.x,
			img_object.bbox.y, img_object.bbox.x + img_object.bbox.width, img_object.bbox.y + img_object.bbox.height, color_img_clone);
		emit(img_ready(color_img_clone, depth_frame));
		color_img_clone.release();

		// 保存图像
		// 获取当前时间点
		auto now = std::chrono::system_clock::now();
		// 获取自纪元（1970-01-01）以来的毫秒数
		auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

		// 判断目标是否走到中心，true保存两次，false保存一次
	/*	cv::Point view_center(img_object.img.cols / 2, img_object.img.rows / 2);
		cv::Point bbox_center(img_object.bbox.x + img_object.bbox.width / 2, img_object.bbox.y + img_object.bbox.height / 2);*/
		//if (cv::norm(view_center - bbox_center) < 200 && !current_record_cow.saved) {
		if (((img_object.bbox.x + img_object.bbox.width / 2) >= img_object.img.cols / 2) && !current_record_cow.saved) {
			std::string show_path =
				root_dir_path + "show_data/"
				+ std::to_string(timestamp) + "_cow" +
				std::to_string(current_record_cow.cow_index);
			current_record_cow.saved = true;
			std::thread([=]() {
				save_all_data(show_path, true, img_object.img, depth_frame);
				}).detach();
		}
		std::string save_path =
			root_dir_path + "saved_data/"
			+ std::to_string(timestamp) + "_cow" +
			std::to_string(current_record_cow.cow_index);
		std::thread([=]() {
			save_all_data(save_path, false, img_object.img, depth_frame);
			}).detach();
		depth_frame.release();
	}
}

