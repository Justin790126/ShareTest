#include "ModelSDK.h"

ModelSDK::ModelSDK(QObject *parent)
    : QThread(parent)
{
}

void ModelSDK::run()
{
    cout << "run" << endl;
}