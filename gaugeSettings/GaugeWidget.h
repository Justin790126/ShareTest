#ifndef GAUGEWIDGET_H
#define GAUGEWIDGET_H

#include <QWidget>
#include <QTableWidget>
#include <QVBoxLayout>
#include <QToolButton>
#include <QListWidget>
#include <QLabel>
#include <QPushButton>


class GaugeWidget : public QWidget {
    Q_OBJECT

public:
    GaugeWidget(QWidget *parent = 0);

private:
    QTableWidget *tableWidget;
    QWidget* settingPage;

private slots:
    void handleToggleSettingWidget();
};

#endif // GAUGEWIDGET_H
