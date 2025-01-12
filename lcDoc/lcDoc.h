#ifndef LC_DOC_H
#define LC_DOC_H

#include <iostream>
#include <vector>
#include <string>
#include <map>

#include <QApplication>
#include <QWidget>
#include "ModelMdReader.h"
#include "ViewManual.h"

class lcDoc : public QWidget
{
    Q_OBJECT
public:
    ModelMdReader *model;
    ViewManual *view;

    lcDoc(QWidget *parent = nullptr);
    ~lcDoc();
    void closeEvent(QCloseEvent *event) override;
    // other UI elements and methods
private slots:
    void handleSearchTextChanged(const QString &msg);
    void handleButtonClick();

    void handleSearchResultClicked(QListWidgetItem *item);
};

#endif // LC_DOC_H