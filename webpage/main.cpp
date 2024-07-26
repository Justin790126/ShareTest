#include <QApplication>
#include "webpage.h"

int main(int argc, char *argv[]) {
    QApplication a(argc, argv);
    Webpage drawingWidget;
    drawingWidget.resize(400, 300);
    drawingWidget.show();
    return a.exec();
}
