#ifndef LC_OVL_PRODUCT_H
#define LC_OVL_PRODUCT_H

#include <iostream>

using namespace std;

#include "ModelOvlConf.h"
#include "ViewProductDialog.h"

class lcOvlProduct : public QObject
{
    Q_OBJECT
    public:
        lcOvlProduct();
        ~lcOvlProduct();

    private:
        ViewProductDialog * view = NULL;
        ModelOvlConf * model = NULL;

    
    private slots:
        void handelRcsvReaded();
};

#endif /* LC_OVL_PRODUCT_H */