// example of qt application

#include <QApplication>
#include "ModelNpz.h"
#include <QtGui>
int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    ModelNpz* npz = new ModelNpz;
    npz->SetFileName("lena.npz");
    npz->start();
    npz->wait(); // Wait for the thread to finish

    double* img = npz->GetImage();
    int w = npz->GetWidth();
    int h = npz->GetHeight();
    int c = npz->GetChannels();

    // draw img with QImage on a widget
    QWidget* wid = new QWidget;
    wid->setWindowTitle("Npz Image Viewer");
    wid->resize(w, h);
    wid->show();

    QImage image(w, h, QImage::Format_RGB888);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int index = (y * w + x) * c;
            int r = static_cast<int>(img[index] * 255);
            int g = static_cast<int>(img[index + 1] * 255);
            int b = static_cast<int>(img[index + 2] * 255);
            image.setPixel(x, y, qRgb(r, g, b));
        }
    }
    QLabel* label = new QLabel(wid);
    label->setPixmap(QPixmap::fromImage(image));
    label->setAlignment(Qt::AlignCenter);
    wid->setLayout(new QVBoxLayout);
    wid->layout()->addWidget(label);

    return app.exec();
}