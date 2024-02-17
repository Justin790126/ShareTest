#include <QtGui>

class CustomTreeWidgetItem : public QTreeWidgetItem
{
public:
    CustomTreeWidgetItem(QTreeWidget *parent = NULL) : QTreeWidgetItem(parent) {}


    void setComboBox(int column, QComboBox *comboBox)
    {
        treeWidget()->setItemWidget(this, column, comboBox);
    }
};

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    QTreeWidget treeWidget;
    treeWidget.setColumnCount(3); // Increase column count

    // Add custom item
    CustomTreeWidgetItem *item = new CustomTreeWidgetItem(&treeWidget);
    item->setText(0, "Item 1");

    // Create and set QComboBox for the second column of the item
    QComboBox *comboBox = new QComboBox(&treeWidget);
    comboBox->addItem("Option 1");
    comboBox->addItem("Option 2");
    item->setComboBox(1, comboBox);

    QCheckBox *checkBox = new QCheckBox(&treeWidget);
    treeWidget.setItemWidget(item, 2, checkBox);

    treeWidget.show();

    return app.exec();
}
