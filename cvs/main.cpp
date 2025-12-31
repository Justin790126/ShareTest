#include <QApplication>
#include "MainWindow.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    MainWindow w;
    w.resize(900, 600);
    w.setWindowTitle("Qt4.8 BBox Gesture in Canvas, RubberBand in MainWindow (Mode-based)");
    w.show();

    return app.exec();
}
