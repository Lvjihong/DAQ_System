#pragma once
#include <QMainWindow>
#include <QThread>
#include <QMutex>
#include <QWaitCondition>
#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>
#include <QImage>
#include <QPixmap>
#include <opencv2/opencv.hpp>
#include <k4a/k4a.hpp>



class KinectWorker : public QThread {
	Q_OBJECT
public:
	KinectWorker() : recording(false) {}
	void run() override;

signals:
	void imageReady(const cv::Mat& rgbMat, const cv::Mat& depthMat);

public slots:
	void startRecording();
	void stopRecording();

private:
	QMutex mutex;
	bool recording;
	cv::VideoWriter rgb_writer;
	cv::VideoWriter depth_writer;
	std::queue<cv::Mat> rgb_queue;
	std::queue<cv::Mat> depth_queue;
	QWaitCondition queue_cond;
};
