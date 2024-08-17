#include "KinectWorker.h"


void KinectWorker::startRecording() {
	const QString rgbPath;
	const QString depthPath;
	QMutexLocker locker(&mutex);
	rgb_writer.open(rgbPath.toStdString(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(1920, 1080));
	depth_writer.open(depthPath.toStdString(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(640, 576));
	recording = true;
}

void KinectWorker::stopRecording() {
	QMutexLocker locker(&mutex);
	recording = false;
	rgb_writer.release();
	depth_writer.release();
}

void KinectWorker::run() {
	k4a::device device = k4a::device::open(K4A_DEVICE_DEFAULT);
	k4a_device_configuration_t config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
	config.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
	config.color_resolution = K4A_COLOR_RESOLUTION_1080P;
	config.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;
	config.camera_fps = K4A_FRAMES_PER_SECOND_30;
	device.start_cameras(&config);

	while (true) {
		k4a::capture capture;
		if (device.get_capture(&capture)) {
			k4a::image rgb_image = capture.get_color_image();
			cv::Mat rgb_mat(cv::Size(1920, 1080), CV_8UC4, (void*)rgb_image.get_buffer(), cv::Mat::AUTO_STEP);

			k4a::image depth_image = capture.get_depth_image();
			cv::Mat depth_mat(cv::Size(640, 576), CV_16U, (void*)depth_image.get_buffer(), cv::Mat::AUTO_STEP);
			cv::Mat depth_mat_8u;
			depth_mat.convertTo(depth_mat_8u, CV_8U, 255.0 / 10000);

			// 克隆图像以避免覆盖
			{
				QMutexLocker locker(&mutex);
				rgb_queue.push(rgb_mat.clone());
				cv::Mat depth_colormap;
				cv::applyColorMap(depth_mat_8u, depth_colormap, cv::COLORMAP_JET);
				depth_queue.push(depth_colormap);
				locker.unlock();
			}
			queue_cond.wakeOne();

			emit imageReady(rgb_mat, depth_mat_8u);

			cv::Mat rgb_frame, depth_frame;
			{
				// 从队列中取出图像数据
				QMutexLocker locker(&mutex);

				while (rgb_queue.empty() || depth_queue.empty()) {
					queue_cond.wait(&mutex);  // 释放mutex并等待，直到收到信号
				}
				rgb_frame = rgb_queue.front();
				depth_frame = depth_queue.front();
				rgb_queue.pop();
				depth_queue.pop();
				locker.unlock();
			}

			// 保存线程处理
			if (!rgb_frame.empty() && !depth_frame.empty()) {
				rgb_writer.write(rgb_frame);
				depth_writer.write(depth_frame);
			}

		}
		else {
			break;
		}
	}

	device.close();
}