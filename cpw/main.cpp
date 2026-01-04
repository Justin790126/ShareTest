#include <QtGui/QApplication>
#include "lcMainWindow.h"

int main(int argc, char *argv[]) {
    QApplication a(argc, argv);
    lcMainWindow controller;
    controller.generateAndLoad(6000); // 模擬載入 6000 筆資料
    return a.exec();
}