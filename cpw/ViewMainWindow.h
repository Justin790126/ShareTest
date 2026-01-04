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
    virtual ~ViewMainWindow();

    // 暴露元件給 Controller 使用
    QTreeView       *treeView;
    QGraphicsView   *graphicsView;
    QGraphicsScene  *scene;

private:
    void setupUi(); // 初始化 UI 佈局
    QSplitter       *m_splitter;
};

#endif // VIEWMAINWINDOW_H