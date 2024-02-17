#include <QtGui>

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    QMainWindow mainWindow;
    mainWindow.setWindowTitle("Main Window");

    // Create central widget for the main window
    QWidget *centralWidget = new QWidget(&mainWindow);
    mainWindow.setCentralWidget(centralWidget);

    // Create a layout for the central widget
    QVBoxLayout *layout = new QVBoxLayout(centralWidget);

    // Add a label to the central widget
    QLabel *label = new QLabel("Central Widget");
    label->setAlignment(Qt::AlignCenter);
    layout->addWidget(label);

    // Create a toolbar
    QToolBar *toolBar = new QToolBar("Tool Bar", &mainWindow);
    mainWindow.addToolBar(toolBar);
    QWidget *toolBarWidget = new QWidget;
    toolBar->addWidget(toolBarWidget);

    // Create a grid layout for the toolbar widget
    QGridLayout *toolBarLayout = new QGridLayout(toolBarWidget);

    // Add buttons to the toolbar layout
    for (int row = 0; row < 2; ++row) {
        for (int col = 0; col < 4; ++col) {
            QToolButton *button = new QToolButton;
            button->setText(QString("Button %1").arg(row * 4 + col + 1));
            toolBarLayout->addWidget(button, row, col);
        }
    }

    // Create a dock widget
    QDockWidget *dockWidget = new QDockWidget("Dock Widget", &mainWindow);
    dockWidget->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);
    dockWidget->setWidget(new QTextEdit);
    mainWindow.addDockWidget(Qt::LeftDockWidgetArea, dockWidget);

    // Create a status bar
    QStatusBar *statusBar = new QStatusBar(&mainWindow);
    mainWindow.setStatusBar(statusBar);
    statusBar->showMessage("Status Bar");

    // Set central widget layout
    centralWidget->setLayout(layout);

    mainWindow.show();

    return app.exec();
}
