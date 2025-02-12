#ifndef LC_SDK_H
#define LC_SDK_H

#include <QtGui>
#include "ModelSDK.h"
#include "TestWidget.h"
#include "ModelSktClnt.h"

class lcSDK : public QObject
{
    Q_OBJECT
    public:
    lcSDK(QObject *parent = 0);
    ~lcSDK() = default;

    private:
    ModelSDK* model;
    TestWidget* view;

    private slots:
        void handleSendMsg();

    private:
        ModelSktClnt* m_clnt = NULL;
};


#endif /* LC_SDK_H */