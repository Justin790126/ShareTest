// example of qt application

#include <QApplication>
#include "ModelNpz.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    ModelNpz* npz = new ModelNpz;
    npz->SetFileName("lena.npz");
    npz->start();

    return app.exec();
}