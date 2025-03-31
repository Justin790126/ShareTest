#ifndef VIEW_GAUGE_SETUP_DIALOG_H
#define VIEW_GAUGE_SETUP_DIALOG_H

#include <QtGui>
#include <iostream>

class ViewGaugeSetupDialog : public QDialog
{
    Q_OBJECT
public:
    ViewGaugeSetupDialog(QWidget *parent = NULL);
    ~ViewGaugeSetupDialog() = default;

    QFrame *CreateSeparator();
    QFrame *CreateVerticalSeparator();

private:
    void Widgets();
    void Layout();
    void UI();
    void Connect();
private:
    QCheckBox* chbNeglectWhiteSpace = NULL;
    QButtonGroup* bgDelimiters = NULL;
    QLineEdit* leOtherDelimiter = NULL;

    QWidget* CreatedDelimiterSetupWidget();

    QPushButton* btnCancel = NULL;
    QPushButton* btnOk = NULL;

};

#endif /* VIEW_GAUGE_SETUP_DIALOG_H */