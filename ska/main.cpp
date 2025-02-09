#include <QtGui>
#include <iostream>
#include "TestWidget.h"

using namespace std;

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    TestWidget w;
    w.show();
    return a.exec();
}
