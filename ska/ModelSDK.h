#ifndef MODEL_SDK_H
#define MODEL_SDK_H

#include <QThread>
#include <iostream>

using namespace std;

class ModelSDK : public QThread
{
    Q_OBJECT
    public:
    ModelSDK(QObject *parent = 0);
    ~ModelSDK() = default;
    protected:
    virtual void run() override;
};

#endif /* MODEL_SDK_H */