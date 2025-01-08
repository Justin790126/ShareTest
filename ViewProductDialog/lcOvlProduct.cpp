#include "lcOvlProduct.h"

lcOvlProduct::lcOvlProduct()
{

    model = new ModelOvlConf();
    // model->SetFname("rcsv.txt");
    // connect(model, SIGNAL(allPageReaded()), this, SLOT(handelRcsvReaded()));
    // model->start();
    // ModelOvlConf *cf = (ModelOvlConf*)QObject::sender();
    // cout << "handleRcsv: " << cf->GetFname()  << endl;

    view = new ViewProductDialog();
    connect(view, SIGNAL(addNewProduct()), this, SLOT(handleAddNewProduct()));
    connect(view, SIGNAL(loadConfig()), this, SLOT(handleLoadOvlConfig()));
    connect(view, SIGNAL(saveConfig()), this, SLOT(handleSaveOvlConfig()));
    connect(view, SIGNAL(delSelProduct()), this, SLOT(handleDelSelProduct()));
    if (view->exec())
    {
        cout << "accept" << endl;
    }
}

lcOvlProduct::~lcOvlProduct()
{
}


void lcOvlProduct::handleSaveOvlConfig()
{
    cout << "saveOvlConfig" << endl;
    ViewProductDialog *dialog = (ViewProductDialog *)QObject::sender();
    if (!dialog)
        return;
    QTreeWidget *wid = dialog->GetProductTreeWidget();

    // iterate top count in wid
    int topCount = wid->topLevelItemCount();
    vector<OvlProductInfo> infos;
    infos.reserve(topCount);
    for (int i = 0; i < topCount; i++)
    {
        ProductTreeItem *item = (ProductTreeItem *)wid->topLevelItem(i);
        OvlProductInfo* product = item->GetProductInfo();
        infos.push_back(*product);
    }

    string fname = "rcsv_table.ini";
    ModelOvlConf *cf = model;
    cf->SetWorkerMode((int)OVL_WRITE_CFG);
    cf->SetFname(fname);
    cf->SetProductInfo(infos);
    cf->start();
    cf->Wait();

    // show message box
    QMessageBox::information(dialog, "Save Config", "Config saved successfully!");
}

void lcOvlProduct::handleDelSelProduct()
{
    ViewProductDialog *dialog = (ViewProductDialog *)QObject::sender();
    if (!dialog)
        return;
    QTreeWidget *wid = dialog->GetProductTreeWidget();
    int curSel = wid->currentIndex().row();
    if (curSel >= 0)
    {
        ProductTreeItem *item = (ProductTreeItem *)wid->takeTopLevelItem(curSel);
        delete item;
    }
}

void lcOvlProduct::handleLoadOvlConfig()
{
    ViewProductDialog *dialog = (ViewProductDialog *)QObject::sender();
    if (!dialog)
        return;

    // get open file dialog
    QString fileName = QFileDialog::getOpenFileName(dialog, tr("Open File"), "", tr("Text Files (*.txt)"));
    if (!fileName.isEmpty())
    {
        // load config from file
        ModelOvlConf *cf = model;
        cf->SetWorkerMode((int)OVL_READ_CFG);
        cf->SetFname(fileName.toStdString());
        cf->start();
        cf->Wait();

        vector<OvlProductInfo> *products = cf->GetProductInfo();
        QTreeWidget *wid = dialog->GetProductTreeWidget();

        wid->clear();
        for (size_t i = 0; i < products->size(); i++)
        {
            OvlProductInfo *product = &((*products)[i]);
            QStringList pdInfo;
            pdInfo << QString::fromStdString(product->GetProductName())
                   << QString::number(product->GetDieWidth())
                   << QString::number(product->GetDieHeight())
                   << QString::number(product->GetDieOffsetX())
                   << QString::number(product->GetDieOffsetY());
            ProductTreeItem *it = new ProductTreeItem(pdInfo, wid);
            it->SetProductInfo(product);
        }
    }
}

void lcOvlProduct::handleAddNewProduct()
{
    ViewProductDialog *dialog = (ViewProductDialog *)QObject::sender();
    if (!dialog)
        return;
    QTreeWidget *wid = dialog->GetProductTreeWidget();

    ViewAddProductDialog addProductDialog(view);
    if (addProductDialog.exec())
    {
        cout << "accept" << endl;
        QLineEdit* pdname = addProductDialog.GetProductNameLineEdit();
        QLineEdit* pdw = addProductDialog.GetDieWLineEdit();
        QLineEdit* pdh = addProductDialog.GetDieHLineEdit();
        QLineEdit* pdx = addProductDialog.GetDieOffsetXLineEdit();
        QLineEdit* pdy = addProductDialog.GetDieOffsetYLineEdit();

        string productName = pdname->text().toStdString();
        double dieW = pdw->text().toDouble();
        double dieH = pdh->text().toDouble();
        double dieOffsetX = pdx->text().toDouble();
        double dieOffsetY = pdy->text().toDouble();

        OvlProductInfo* product = model->AddNewProductInfo(productName, dieW, dieH, dieOffsetX, dieOffsetY);
        QStringList pdInfo;
        pdInfo << pdname->text() << pdw->text() << pdh->text() << pdx->text() << pdy->text();
        ProductTreeItem *it = new ProductTreeItem(pdInfo, wid);
        it->SetProductInfo(product);
    }
}

void lcOvlProduct::handelRcsvReaded()
{
}