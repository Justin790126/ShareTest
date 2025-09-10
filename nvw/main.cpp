#include <QtGui>
#include <iostream>
#include "ViewGpuSetup.h"

using namespace std;

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    ViewGpuSetup w;
    w.show();

    return a.exec();
}
