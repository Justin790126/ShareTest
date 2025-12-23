#include <QtGui/QApplication>
#include "view.h"

int main(int argc, char** argv)
{
    QApplication app(argc, argv);

    DemoView w;
    w.resize(900, 650);
    w.show();

    return app.exec();
}
