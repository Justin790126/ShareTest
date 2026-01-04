#ifndef VIEWMAINWINDOW_H
#define VIEWMAINWINDOW_H
#include <QMainWindow>
#include <QTreeView>
#include <QGraphicsView>
#include <QGraphicsScene>
#include <QSplitter>

class ViewMainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit ViewMainWindow(QWidget *parent = 0);
    QTreeView *treeView;
    QGraphicsView *graphicsView;
    QGraphicsScene *scene;
};
#endif