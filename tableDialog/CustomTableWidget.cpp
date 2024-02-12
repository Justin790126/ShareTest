#include "CustomTableWidget.h"
#include <QHeaderView>
#include <QDebug>

CustomTableWidget::CustomTableWidget(QWidget *parent) : QTableWidget(parent) {
    connect(this->verticalHeader(), SIGNAL(sectionPressed(int)), this, SLOT(selectRow(int)));
    connect(this->horizontalHeader(), SIGNAL(sectionPressed(int)), this, SLOT(selectColumn(int)));

    this->horizontalHeader()->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(this->horizontalHeader(), SIGNAL(customContextMenuRequested(const QPoint &)), this, SLOT(showContextMenu(const QPoint &)));

}

CustomTableWidget::CustomTableWidget(int rows, int columns, QWidget *parent) : QTableWidget(rows, columns, parent) {
    connect(this->verticalHeader(), SIGNAL(sectionPressed(int)), this, SLOT(selectRow(int)));
    connect(this->horizontalHeader(), SIGNAL(sectionPressed(int)), this, SLOT(selectColumn(int)));

    this->horizontalHeader()->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(this->horizontalHeader(), SIGNAL(customContextMenuRequested(const QPoint &)), this, SLOT(showContextMenu(const QPoint &)));

}

void CustomTableWidget::showContextMenu(const QPoint &pos) {
        QMenu contextMenu("Context menu", this);
        QAction sortAction("Sort", this);
        connect(&sortAction, SIGNAL(triggered()), this, SLOT(sortColumn()));
        contextMenu.addAction(&sortAction);
        contextMenu.exec(pos);
    }

void CustomTableWidget::mousePressEvent(QMouseEvent *event) {
    // Call base class implementation for other mouse events
    QTableWidget::mousePressEvent(event);

    // If no items are selected, clear the selection
    if (selectedItems().isEmpty()) {
        clearSelection();
    }
}

void CustomTableWidget::selectRowByHeader(int row) {
    this->selectRow(row);
}

void CustomTableWidget::selectColumnByHeader(int column) {
    this->selectColumn(column);
}

void CustomTableWidget::sortColumn() {
    int column = this->currentColumn();
    this->sortByColumn(column, Qt::AscendingOrder);
}