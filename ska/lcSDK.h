#ifndef LC_SDK_H
#define LC_SDK_H

#include <QtGui>
#include "ModelSDK.h"
#include "TestWidget.h"

class lcSDK : public QObject
{
    Q_OBJECT
    public:
    lcSDK(QObject *parent = 0);
    ~lcSDK() = default;

    private:
    ModelSDK* model;
    TestWidget* view;
};


#endif /* LC_SDK_H */