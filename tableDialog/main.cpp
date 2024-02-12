#include <QtGui>
#include "ViewTableDialog.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    ViewTableDialog dialog;
    dialog.setWindowTitle("Custom Dialog");
    dialog.resize(600, 400);
    dialog.exec();

    return app.exec();
}
