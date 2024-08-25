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
#include <opencv2/tracking.hpp>
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

#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <QCloseEvent>
#include <QMessageBox>
#include <QTimer>
#include <QThread>
#include <QMutex>
#include <QWaitCondition>
#include <k4a/k4a.hpp>

#include "MultiDeviceCapturer.h"
#include "YOLOv8.h"
#include "tinyxml.h"

struct cow {
	int cow_index;
	float center_x;
	float center_y;
	cv::Rect bbox;
	bool saved;
	float confidence;
};
struct color_img_with_bbox {
	cv::Mat img;
	cv::Rect bbox;
	float confidence;
};
class DAQ_System : public QWidget {
	Q_OBJECT

public:
	DAQ_System(QWidget* parent = nullptr);
	~DAQ_System();

	void init_cameras();
	// 读取基本参数
	void xmlparse(vector<std::uint32_t>& device_indices, int32_t& exposure_time,
		int& color_res, int& dep_mode, int& fps, std::string& root_dir

	);
	void xml_parse(int& color_res, int& dep_mode, int& exposure, int& fps, std::string& save_dir, double& threshold_distance, int& threshold_bbox_width, int& threshold_bbox_height);

	void xml_set(k4a::calibration cali);
	cv::Mat color_to_opencv(const k4a::image& im);
	cv::Mat depth_to_opencv(const k4a::image& im);
	k4a_device_configuration_t get_master_config(int color_res, int dep_mode,
		int fps);
	k4a_device_configuration_t get_subordinate_config(int color_res, int dep_mode,
		int fps);
	k4a::image create_depth_image_like(const k4a::image& im);
	k4a::image create_depth_image_like(int w, int h);
	k4a_device_configuration_t DAQ_System::get_default_config(int color_res,
		int dep_mode,
		int fps);
	QImage opencv_to_QImage(cv::Mat cvImg);
	double calculateIoU(const cv::Rect& rect1, const cv::Rect& rect2);
	double calculate_distance_of_centers_bbox(const cv::Rect& rect1, const cv::Rect& rect2);
	void save_all_data(const std::string save_dir, const bool need_show,
		const k4a::capture capture);
	void save_all_data(const std::string save_path, const bool need_show,
		const cv::Mat color_img, const cv::Mat depth_img);

	void image_capture_thread();
	void image_handle_thread();
	void image_save_thread();
	void startThreads();

protected:
	void closeEvent(QCloseEvent* event) override;
signals:
	void showPseudoColorImg(cv::Mat);
	void img_ready(cv::Mat img_rgb, cv::Mat img_depth);

	void save_data(const cv::Point& center, const int cow_index,
		const k4a::capture capture);
	void newImageCaptured();
	void imageProcessed();


public slots:
	void on_startButton_clicked();
	void on_captureButton_clicked(const cv::Point& center, const int cow_index,
		const k4a::capture capture);
	void on_stopButton_clicked();
	void updateFrame();
	void showImg(cv::Mat img_rgb, cv::Mat img_depth);

private:
	Ui::DAQ_SystemClass ui;
	const uint32_t MIN_TIME_BETWEEN_DEPTH_CAMERA_PICTURES_USEC = 160;
	const double THRESHOLD_IOU = 0.5;
	double THRESHOLD_DISTANCE = 400;
	int THRESHOLD_BBOX_WIDTH = 600;
	int THRESHOLD_BBOX_HEIGHT = 350;
	bool isCameraRunning = false;  //是否相机开启
	int cow_index = 0;             //经过牛的索引
	std::string root_dir_path = "./data/";
	std::string saved_path = root_dir_path + "saved_path/";

	QTimer* timer;

	k4a::device device;
	k4a::capture capture;
	k4a_device_configuration_t config;
	size_t num_devices = 1;
	k4a::calibration cali;
	Yolov8 yolo;
	cv::dnn::Net net;
	cv::Ptr<cv::Tracker> tracker;
	int count_detect = 0;
	int count = 0;
	bool need_detect = true;
	cow current_record_cow;


	// 多线程相关
	std::queue<cv::Mat> rgb_queue;
	std::queue<cv::Mat> depth_queue;
	std::queue<color_img_with_bbox> img_with_bbox_queue;
	std::mutex rgb_mutex;
	std::mutex depth_mutex;
	std::mutex bbox_mutex;

	std::condition_variable rgb_condition;
	std::condition_variable depth_condition;
	std::condition_variable bbox_condition;

	std::thread capture_thread;
	std::thread handle_thread;
	std::thread save_thread;
};

