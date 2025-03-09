#include <QtGui>
#include <QApplication>
#include <QMainWindow>
#include <QVBoxLayout>
#include "qcustomplot.h"
#include <vector>
#include <random>
#include <cmath>

// Define the NcePoint structure
struct NcePoint {
    double x, y, z;
};

// Function to generate Jet colormap color based on a normalized value (0 to 1)
QColor getJetColor(double value) {
    value = std::max(0.0, std::min(1.0, value));

    double r, g, b;
    if (value < 0.25) {
        r = 0;
        g = 4 * value;
        b = 1;
    } else if (value < 0.5) {
        r = 0;
        g = 1;
        b = 1 - 4 * (value - 0.25);
    } else if (value < 0.75) {
        r = 4 * (value - 0.5);
        g = 1;
        b = 0;
    } else {
        r = 1;
        g = 1 - 4 * (value - 0.75);
        b = 0;
    }

    return QColor(static_cast<int>(r * 255), static_cast<int>(g * 255), static_cast<int>(b * 255));
}

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    // Create a main window
    QMainWindow window;
    QWidget* centralWidget = new QWidget(&window);
    QVBoxLayout* layout = new QVBoxLayout(centralWidget);
    window.setCentralWidget(centralWidget);

    // Initialize QCustomPlot
    QCustomPlot* customPlot = new QCustomPlot(centralWidget);
    layout->addWidget(customPlot);

    // Generate random data
    std::vector<NcePoint> points(1024);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-10.0, 10.0); // Range for x, y
    std::uniform_real_distribution<> zDis(0.0, 1.0);   // Range for z (0 to 1 for colormap)

    double minZ = std::numeric_limits<double>::max();
    double maxZ = std::numeric_limits<double>::lowest();

    for (auto& point : points) {
        point.x = dis(gen);
        point.y = dis(gen);
        point.z = zDis(gen);
        minZ = std::min(minZ, point.z);
        maxZ = std::max(maxZ, point.z);
    }

    // Create scatter points using QCPItemEllipse
    for (const auto& point : points) {
        QCPItemEllipse* ellipse = new QCPItemEllipse(customPlot);
        double normalizedZ = (point.z - minZ) / (maxZ - minZ); // Normalize z to [0, 1]
        QColor color = getJetColor(normalizedZ);

        // Set position and size of the ellipse (small circle for scatter point)
        double radius = 0.05; // Size of the point
        ellipse->topLeft->setCoords(point.x - radius, point.y + radius);
        ellipse->bottomRight->setCoords(point.x + radius, point.y - radius);

        // Set color
        ellipse->setPen(QPen(color));
        ellipse->setBrush(QBrush(color));
    }

    // Set axis labels and ranges
    customPlot->xAxis->setLabel("X");
    customPlot->yAxis->setLabel("Y");
    customPlot->xAxis->setRange(-12, 12);
    customPlot->yAxis->setRange(-12, 12);

    // Add a color scale (legend) for reference
    QCPColorScale* colorScale = new QCPColorScale(customPlot);
    customPlot->plotLayout()->addElement(0, 1, colorScale); // Add to right of plot
    colorScale->setType(QCPAxis::atRight);
    QCPColorGradient gradient(QCPColorGradient::gpJet);
    colorScale->setGradient(gradient);
    colorScale->setDataRange(QCPRange(minZ, maxZ));
    colorScale->setLabel("Z Value");

    // Enable interactions (optional)
    customPlot->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom);

    // Adjust layout and display
    customPlot->rescaleAxes();
    window.resize(800, 600);
    window.show();

    return app.exec();
}