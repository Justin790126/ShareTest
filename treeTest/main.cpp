#include <iostream>
#include <random>

#include <QWidget>
#include <QPainter>
#include <QMainWindow>
#include <QApplication>
#include <QVBoxLayout>
#include <QPushButton>
#include "treeWidget.h"
class MainWindow : public QMainWindow {

public:
    MainWindow(QWidget* parent = NULL) : QMainWindow(parent) {
        setWindowTitle("Tree Visualization");

        treeWidget = new TreeWidget(this);

        QWidget* cen = new QWidget;
        btnUpdate = new QPushButton("Update");
        btnUpdate->setFixedHeight(20);
        QVBoxLayout* layout = new QVBoxLayout;
        layout->addWidget(treeWidget);
        layout->addWidget(btnUpdate);
        layout->setContentsMargins(0,0,0,0);
        cen->setLayout(layout);
        setCentralWidget(cen);


        connect(btnUpdate, SIGNAL(clicked()), treeWidget, SLOT(update()));
    }
    QPushButton* btnUpdate;

private:
    TreeWidget* treeWidget;

};

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    
    // Perform an in-order traversal to print nodes
    MainWindow mainWindow;
    mainWindow.show();

    return app.exec();
}
