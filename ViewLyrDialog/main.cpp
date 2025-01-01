#include <QtGui>
#include <iostream>
#include "ViewLyrDialog.h"
using namespace std;

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    ViewLyrDialog dialog;
    dialog.exec();

    return a.exec();
}
