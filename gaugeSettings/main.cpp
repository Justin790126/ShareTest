#include <QApplication>
#include "GaugeWidget.h"

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    // Create an instance of GaugeWidget
    GaugeWidget gaugeWidget;
    gaugeWidget.show();

    return app.exec();
}
