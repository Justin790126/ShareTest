#include <QApplication>
#include <QTreeWidget>
#include <QComboBox>
#include <QLineEdit>
#include <QTreeWidgetItem>
#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QWidget>
#include "TreeWidgetMover.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    QWidget window;
    QVBoxLayout* mainLayout = new QVBoxLayout(&window);

    QTreeWidget* tree = new QTreeWidget;
    tree->setColumnCount(3);
    QStringList headers;
    headers << "Column 1" << "Column 2" << "Column 3";
    tree->setHeaderLabels(headers);

    // Add 10 rows
    for (int i = 0; i < 10; ++i) {
        QTreeWidgetItem *item = new QTreeWidgetItem(tree);
        item->setText(0, QString("Row %1").arg(i + 1));

        QComboBox *combo = new QComboBox(tree);
        combo->addItems(QStringList() << "Option 1" << "Option 2" << "Option 3");
        tree->setItemWidget(item, 1, combo);

        QLineEdit *lineEdit = new QLineEdit(tree);
        lineEdit->setText(QString("LineEdit %1").arg(i + 1));
        tree->setItemWidget(item, 2, lineEdit);
    }

    mainLayout->addWidget(tree);

    QHBoxLayout* btnLayout = new QHBoxLayout;
    QPushButton* upBtn = new QPushButton("Up");
    QPushButton* downBtn = new QPushButton("Down");
    btnLayout->addWidget(upBtn);
    btnLayout->addWidget(downBtn);
    mainLayout->addLayout(btnLayout);

    // Use TreeWidgetMover class to handle movement
    TreeWidgetMover mover(tree, upBtn, downBtn);

    window.setWindowTitle("QTreeWidget Row Up/Down Example");
    window.resize(500, 400);
    window.show();

    return app.exec();
}