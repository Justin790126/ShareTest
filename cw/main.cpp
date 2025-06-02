#include <QtGui>
#include <iostream>
#include "lcChartWizard.h"
#include "ViewScatterStyleCombobox.h"
using namespace std;

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    lcChartWizard wizard;
    // QWidget win;
    // QVBoxLayout* layout = new QVBoxLayout(&win);

    // ViewScatterStyleCombobox* combo = new ViewScatterStyleCombobox();
    // layout->addWidget(combo);

    // win.show();

    return a.exec();
}
