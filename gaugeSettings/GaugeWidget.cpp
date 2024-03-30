#include <QApplication>
#include "GaugeWidget.h"

GaugeWidget::GaugeWidget(QWidget *parent) : QWidget(parent) {
    QHBoxLayout *layout = new QHBoxLayout(this);

    // Left side: Table Widget
    tableWidget = new QTableWidget(this);
    tableWidget->setColumnCount(3);
    tableWidget->setRowCount(5);

    // Right side: Sidebar with Tool Buttons
    QWidget *sidebarWidget = new QWidget(this);
    sidebarWidget->setFixedWidth(60);
    QVBoxLayout *sidebarLayout = new QVBoxLayout(sidebarWidget);
    sidebarLayout->setContentsMargins(0, 0, 0, 0);

    settingPage = new QWidget;
    {
        QVBoxLayout *settingLayout = new QVBoxLayout(settingPage);
        settingLayout->setContentsMargins(0, 0, 0, 0);
        {
            QListWidget* lw = new QListWidget;
            settingLayout->addWidget(lw);
            QHBoxLayout* hlytSettin2 = new QHBoxLayout;
            hlytSettin2->setContentsMargins(0, 0, 0, 0);
            {
                QLabel* lbl = new QLabel("Current Gauge Setting: ");
                hlytSettin2->addWidget(lbl);
                QPushButton* btnUnit = new QPushButton("Unit: nm");
                hlytSettin2->addWidget(btnUnit);
            }
            settingLayout->addLayout(hlytSettin2);
        }
    }
    settingPage->setVisible(false);

    QIcon icons[3] = {QApplication::style()->standardIcon(QStyle::SP_FileIcon),
                      QApplication::style()->standardIcon(QStyle::SP_DialogSaveButton),
                      QApplication::style()->standardIcon(QStyle::SP_DriveCDIcon)};

        QToolButton *toolButton = new QToolButton(sidebarWidget);
        toolButton->setIcon(icons[0]);
        toolButton->setToolButtonStyle(Qt::ToolButtonTextUnderIcon); // Set style to show text under the icon
        toolButton->setText("Settings");
        sidebarLayout->addWidget(toolButton);
        connect(toolButton, SIGNAL(clicked()), this, SLOT(handleToggleSettingWidget()));
    

    // Add stretch to push buttons to the top
    sidebarLayout->addStretch();


    layout->addWidget(tableWidget,6);
    layout->addWidget(settingPage,4);
    layout->addWidget(sidebarWidget);

    setLayout(layout);
}


void GaugeWidget::handleToggleSettingWidget() {
    if (settingPage->isVisible()) {
        settingPage->setVisible(false);
    } else {
        settingPage->setVisible(true);
    }
}