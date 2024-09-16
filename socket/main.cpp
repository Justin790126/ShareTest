#include <QApplication>
#include <QWidget>
#include <QLabel>
#include <QPushButton>


int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    QWidget window;
    window.setWindowTitle("My Qt Application");
    window.resize(300, 200);


    QLabel label(&window);
    label.setText("Hello, Qt!");
    label.move(100, 50);

    window.show();

    return app.exec();
}