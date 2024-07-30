#include "DAQ_System.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    DAQ_System* w = new DAQ_System;
    w->setWindowState(Qt::WindowMaximized);
    w->show();
    return a.exec();
}
