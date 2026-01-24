#include <QApplication>
#include "ViewBoolTableWidget.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    ViewBoolTableWidget w;
    w.resize(980, 300);
    w.show();

    return app.exec();
}
