#include <QApplication>
#include "BoolTableWidget.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    BoolTableWidget w;
    w.resize(800, 300);
    w.show();

    return app.exec();
}
