// mainwindow.cpp
#include "mainwindow.h"


MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    m_scene = new QGraphicsScene(this);
    m_view = new QGraphicsView(m_scene);
    setCentralWidget(m_view);

    // QList<QPointF> vertices;
    // vertices << QPointF(0, -50) << QPointF(43.3, -25) << QPointF(43.3, 25)
    //          << QPointF(0, 50) << QPointF(-43.3, 25) << QPointF(-43.3, -25);
    // HexagonItem* hexagonItem = new HexagonItem(vertices);
    // hexagonItem->setPos(100, 100); // Set position
    // m_scene->addItem(hexagonItem);
    QList<QPointF> vertices;
    vertices << QPointF(0, -50) << QPointF(43.3, -25) << QPointF(43.3, 25)
             << QPointF(0, 50) << QPointF(-43.3, 25) << QPointF(-43.3, -25);

    // Create a HexagonItem instance and add it to the scene
    HexagonItem* hexagonItem = new HexagonItem(vertices);
    m_scene->addItem(hexagonItem);
}

MainWindow::~MainWindow()
{

}
