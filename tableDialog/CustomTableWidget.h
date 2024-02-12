#ifndef CUSTOMTABLEWIDGET_H
#define CUSTOMTABLEWIDGET_H

#include <QTableWidget>
#include <QMouseEvent>
#include <QMenu>

class CustomTableWidget : public QTableWidget {
    Q_OBJECT

public:
    CustomTableWidget(QWidget *parent = NULL);
    CustomTableWidget(int rows, int columns, QWidget *parent = 0);
protected:
    void mousePressEvent(QMouseEvent *event) override;

public slots:
    void selectRowByHeader(int logicalIndex);
    void selectColumnByHeader(int logicalIndex);
    void showContextMenu(const QPoint &pos);
    void sortColumn();
};

#endif // CUSTOMTABLEWIDGET_H
