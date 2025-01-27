#include <QApplication>
#include "lcTensorBoard.h"
int main(int argc, char** argv)
{
    QApplication app(argc, argv);
    LcTensorBoard tb;
    return app.exec();
}