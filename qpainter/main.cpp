#include <QApplication>
#include <QWidget>
#include <QPainter>
#include <QPixmap>
#include <QPen>
#include <QtGui>
#include <QtCore>

#include "MainWidget.h"

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    // MyWidget widget;
    // widget.resize(300, 400);
    // widget.show();
    MainWidget* wid = new MainWidget;
    wid->resize(640,480);
    

    
    wid->show();

    return app.exec();
}
