#include "lcOvlProduct.h"


lcOvlProduct::lcOvlProduct() {

    model = new ModelOvlConf();
    model->SetFname("rcsv.txt");
    connect(model, SIGNAL(allPageReaded()), this, SLOT(handelRcsvReaded()));
    model->start();
  
}

lcOvlProduct::~lcOvlProduct() {

}

void lcOvlProduct::handleAddNewProduct() {

    ViewAddProductDialog addProductDialog();
    if (addProductDialog.exec()) {
        cout << "accept" << endl;
    }
}

void lcOvlProduct::handelRcsvReaded() {

    ModelOvlConf *cf = (ModelOvlConf*)QObject::sender();
    cout << "handleRcsv: " << cf->GetFname()  << endl;
      
    view = new ViewProductDialog();
    QPushButton *btnAdd = view->GetAddButton();
    connect(btnAdd, SIGNAL(clicked()), this, SLOT(handleAddNewProduct()));
    if (view->exec()) {
        cout << "accept" << endl;
    }
}