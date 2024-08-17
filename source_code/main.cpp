#include "DAQ_System.h"
#include <QtWidgets/QApplication>
#include <QtCore/qmetatype.h>
int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    qRegisterMetaType<cv::Mat>("cv::Mat");
    DAQ_System* w = new DAQ_System;
    w->setWindowState(Qt::WindowMaximized);
    w->show();
    return a.exec();
}
