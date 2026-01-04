#include <QtGui/QApplication>
#include <QMetaType>
#include <QVector>
#include "lcMainWindow.h"

int main(int argc, char *argv[]) {
    // 1. 初始化 Qt GUI 應用程式
    QApplication a(argc, argv);

    // 2. 實例化控制器 (Controller)
    // 控制器會在建構子中自動建立 View 與 Model，並建立連結
    lcMainWindow controller;

    // 3. 載入二進位檔
    // 這裡預設載入同目錄下由 gen_prof 產生的 test.cellprofile
    // ModelCellProfile (QThread) 會在背景開始解析，不影響介面反應
    controller.loadProfile("test.cellprofile");

    // 4. 進入事件迴圈
    return a.exec();
}