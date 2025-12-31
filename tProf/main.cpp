#include <QApplication>
#include "MainWindow.h"

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);

    MainWindow w;

    if (argc >= 2) {
        // 若從命令列給檔案路徑，直接塞進欄位（使用者按 Parse 即可）
        // 你也可改成自動 parse
        // w.setPath(argv[1]);  // 若你想做這功能，我可以幫你加
    }

    w.show();
    return app.exec();
}
