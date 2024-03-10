#include <QApplication>
#include "drawingwidget.h"

int main(int argc, char *argv[]) {
    QApplication a(argc, argv);
    DrawingWidget drawingWidget;
    drawingWidget.resize(400, 300);
    drawingWidget.show();
    return a.exec();
}
