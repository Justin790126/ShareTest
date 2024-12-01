
#include <QApplication>
#include <QWidget>
#include "ChartWizard.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    ChartWizard* chart = new ChartWizard;
    // chart->show();
    return a.exec();
}
