#include <QApplication>
#include "CustomTableWidget.h"
#include "CustomTableWidgetItem.h"
#include <QHeaderView>
#include <QMenu>
#include <QAction>

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    
    CustomTableWidget table(5, 5);
    table.setItem(0, 0, new CustomTableWidgetItem("5"));
    table.setItem(1, 0, new CustomTableWidgetItem("2"));
    table.setItem(2, 0, new CustomTableWidgetItem("10"));
    table.setItem(3, 0, new CustomTableWidgetItem("1"));
    table.setItem(4, 0, new CustomTableWidgetItem("3"));
    
    table.show();

    return app.exec();
}
