#include <QtGui/QApplication>
#include "lcMainWindow.h"

int main(int argc, char *argv[]) {
    QApplication a(argc, argv);
    lcMainWindow controller;
    controller.loadProfile("test.cellprofile");
    return a.exec();
}