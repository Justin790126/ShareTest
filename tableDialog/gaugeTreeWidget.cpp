#include <QtGui>

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    // Create the main window
    QMainWindow mainWindow;
    mainWindow.setWindowTitle("Dock Widget Example");
    mainWindow.resize(800, 600);

    // Create a dock widget
    QDockWidget *dockWidget = new QDockWidget("Dock Widget", &mainWindow);
    dockWidget->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);

    
    QTreeWidget* treeWidget = new QTreeWidget;

    // Set column count (assuming you have 3 columns)
    treeWidget->setColumnCount(3);
    
   
    QWidget* main = new QWidget;
    QVBoxLayout *mainLayout = new QVBoxLayout;
    QHBoxLayout *hLayout = new QHBoxLayout;
    hLayout->setContentsMargins(0, 0, 0, 0);
    QWidget *widgetBtns = new QWidget;
    hLayout->addStretch();
    hLayout->addWidget(new QPushButton("Add"));
    hLayout->addWidget(new QPushButton("Delete"));
    hLayout->addWidget(new QPushButton("Save"));
    widgetBtns->setLayout(hLayout);
    mainLayout->setContentsMargins(0, 0, 0, 0);
    mainLayout->addWidget(treeWidget);
    mainLayout->addWidget(widgetBtns);
    main->setLayout(mainLayout);



    dockWidget->setWidget(main);
    mainWindow.show();

    return app.exec();
}
