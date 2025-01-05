#include <QtGui>
#include <iostream>
#include "lcOvlProduct.h"

using namespace std;

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    lcOvlProduct* pd = new lcOvlProduct();

    return a.exec();
}
