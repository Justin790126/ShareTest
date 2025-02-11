
#include "lcSDK.h"

lcSDK::lcSDK(QObject *parent): QObject(parent)
{
    model = new ModelSDK();
    view = new TestWidget();
    model->start();
    view->show();
}
