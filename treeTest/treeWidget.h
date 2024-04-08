
#include <QWidget>
#include <QPainter>
#include <QMainWindow>
#include <QApplication>
#include <QVBoxLayout>
#include <QPushButton>
#include <QImage>

#include "tree.h"


// Define TreeNode and Tree classes as before...

class TreeWidget : public QWidget {

    Q_OBJECT
public:
    TreeWidget(QWidget* parent = NULL);

public slots:
    void update();


protected:
    void paintEvent(QPaintEvent* event);

private:
    Tree<int> tree;
    QImage img;
};