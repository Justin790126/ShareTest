#ifndef TREEWIDGETMOVER_H
#define TREEWIDGETMOVER_H

#include <QObject>
#include <QTreeWidget>
#include <QPushButton>

class TreeWidgetMover : public QObject
{
    Q_OBJECT
public:
    TreeWidgetMover(QTreeWidget* tree, QPushButton* upBtn, QPushButton* downBtn, QObject* parent = nullptr);

public slots:
    void moveUp();
    void moveDown();

private:
    void moveItemWithFreshWidgets(int from, int to);

    QTreeWidget* m_tree;
    QPushButton* m_upBtn;
    QPushButton* m_downBtn;
};

#endif // TREEWIDGETMOVER_H