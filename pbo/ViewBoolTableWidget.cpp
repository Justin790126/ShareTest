#include "ViewBoolTableWidget.h"
#include "ViewPathPickerWidget.h"
#include "ViewOperationWidget.h"

#include <QTableWidget>
#include <QHeaderView>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QMenu>
#include <QMessageBox>
#include <QAction>

ViewBoolTableWidget::ViewBoolTableWidget(QWidget* parent)
    : QWidget(parent)
    , m_table(0)
    , m_contextRow(-1)
    , m_undoValid(false)
    , m_undoRowIndex(-1)
{
    initTable();

    // Bottom buttons: Cancel LEFT, OK RIGHT
    QPushButton* cancelBtn = new QPushButton("Cancel", this);
    QPushButton* okBtn = new QPushButton("OK", this);
    connect(okBtn, SIGNAL(clicked()), this, SLOT(onOk()));
    connect(cancelBtn, SIGNAL(clicked()), this, SLOT(onCancel()));

    QHBoxLayout* bottom = new QHBoxLayout;
    bottom->addStretch(1);
    bottom->addWidget(cancelBtn);
    bottom->addWidget(okBtn);

    QVBoxLayout* main = new QVBoxLayout(this);
    main->addWidget(m_table);
    main->addLayout(bottom);

    // Undo shortcut: Ctrl+Z
    QAction* undoAct = new QAction(this);
    undoAct->setShortcut(QKeySequence::Undo);
    addAction(undoAct);
    connect(undoAct, SIGNAL(triggered()), this, SLOT(undoDelete()));
}

void ViewBoolTableWidget::initTable()
{
    m_table = new QTableWidget(this);
    m_table->setColumnCount(3);
    m_table->setHorizontalHeaderLabels(QStringList() << "Input A" << "Operation" << "Input B");

    // show vertical header
    QHeaderView* vh = m_table->verticalHeader();
    vh->show();
    vh->setResizeMode(QHeaderView::Fixed);
    vh->setDefaultSectionSize(78);

    // double-click to add
    connect(vh, SIGNAL(sectionDoubleClicked(int)),
            this, SLOT(onVHeaderDoubleClicked(int)));

    // context menu for delete/undo
    vh->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(vh, SIGNAL(customContextMenuRequested(QPoint)),
            this, SLOT(onVHeaderContextMenu(QPoint)));

    m_table->setSelectionMode(QAbstractItemView::NoSelection);
    m_table->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_table->setFocusPolicy(Qt::NoFocus);

    m_table->horizontalHeader()->setResizeMode(0, QHeaderView::Stretch);
    m_table->horizontalHeader()->setResizeMode(1, QHeaderView::ResizeToContents);
    m_table->horizontalHeader()->setResizeMode(2, QHeaderView::Stretch);
    m_table->setColumnWidth(1, 320);

    // Start with 1 data row + 1 add-row
    m_table->setRowCount(2);
    createRow(0, false);
    createRow(1, true);

    updateRowHeights();
    updateVerticalHeaderLabels();
}

int ViewBoolTableWidget::addRowIndex() const
{
    return m_table->rowCount() - 1;
}

bool ViewBoolTableWidget::isAddRow(int row) const
{
    return row == addRowIndex();
}

void ViewBoolTableWidget::updateRowHeights()
{
    for (int r = 0; r < m_table->rowCount(); ++r)
        m_table->setRowHeight(r, 78);
    m_table->verticalHeader()->setDefaultSectionSize(78);
}

void ViewBoolTableWidget::createRow(int row, bool addRow)
{
    // col0
    ViewPathPickerWidget* left = new ViewPathPickerWidget(m_table);
    left->setDialogTitle("Open input A");
    left->setNameFilter("All Files (*)");
    m_table->setCellWidget(row, 0, left);

    // col1
    ViewOperationWidget* op = new ViewOperationWidget(m_table);
    m_table->setCellWidget(row, 1, op);

    // col2
    ViewPathPickerWidget* right = new ViewPathPickerWidget(m_table);
    right->setDialogTitle("Open input B");
    right->setNameFilter("All Files (*)");
    m_table->setCellWidget(row, 2, right);

    if (addRow)
    {
        // Robust UX: add-row is instruction row, not editable
        left->setPlaceholder("Double-click '+' to add");
        right->setPlaceholder("Double-click '+' to add");

        left->setReadOnly(true);
        right->setReadOnly(true);
        left->setBrowseEnabled(false);
        right->setBrowseEnabled(false);

        op->setEnabled(false);
    }
    else
    {
        left->setPlaceholder("type path or browse...");
        right->setPlaceholder("type path or browse...");

        left->setReadOnly(false);
        right->setReadOnly(false);
        left->setBrowseEnabled(true);
        right->setBrowseEnabled(true);

        op->setEnabled(true);
    }
}

void ViewBoolTableWidget::updateVerticalHeaderLabels()
{
    const int rows = m_table->rowCount();
    for (int r = 0; r < rows; ++r)
    {
        QTableWidgetItem* it = m_table->verticalHeaderItem(r);
        if (!it) it = new QTableWidgetItem();

        if (r == rows - 1) it->setText("+");
        else it->setText(QString::number(r + 1));

        m_table->setVerticalHeaderItem(r, it);
    }
}

void ViewBoolTableWidget::appendDataRow()
{
    // Insert before add-row
    int insertAt = addRowIndex();
    m_table->insertRow(insertAt);

    // new data row
    createRow(insertAt, false);

    // last row must still be add-row (rebuild it for safety)
    createRow(addRowIndex(), true);

    updateRowHeights();
    updateVerticalHeaderLabels();
}

void ViewBoolTableWidget::onVHeaderDoubleClicked(int logicalIndex)
{
    if (logicalIndex < 0) return;
    if (!isAddRow(logicalIndex)) return; // only '+' row

    appendDataRow();

    // Focus new row Input A
    int newRow = addRowIndex() - 1;
    QWidget* w = m_table->cellWidget(newRow, 0);
    if (w) w->setFocus();
}

bool ViewBoolTableWidget::captureRowData(int row, QString& left, QString& op, QString& right) const
{
    if (row < 0 || row >= m_table->rowCount()) return false;
    if (isAddRow(row)) return false;

    ViewPathPickerWidget* l = qobject_cast<ViewPathPickerWidget*>(m_table->cellWidget(row, 0));
    ViewOperationWidget*  o = qobject_cast<ViewOperationWidget*>(m_table->cellWidget(row, 1));
    ViewPathPickerWidget* r = qobject_cast<ViewPathPickerWidget*>(m_table->cellWidget(row, 2));
    if (!l || !o || !r) return false;

    left = l->path();
    op   = o->operation();
    right= r->path();
    return true;
}

void ViewBoolTableWidget::applyRowData(int row, const QString& left, const QString& op, const QString& right)
{
    ViewPathPickerWidget* l = qobject_cast<ViewPathPickerWidget*>(m_table->cellWidget(row, 0));
    ViewOperationWidget*  o = qobject_cast<ViewOperationWidget*>(m_table->cellWidget(row, 1));
    ViewPathPickerWidget* r = qobject_cast<ViewPathPickerWidget*>(m_table->cellWidget(row, 2));
    if (!l || !o || !r) return;

    l->setPath(left);
    o->setOperation(op);
    r->setPath(right);
}

void ViewBoolTableWidget::onVHeaderContextMenu(const QPoint& pos)
{
    int row = m_table->verticalHeader()->logicalIndexAt(pos);
    if (row < 0) return;

    QMenu menu(this);

    // Undo appears if available
    QAction* undoAct = 0;
    if (m_undoValid)
        undoAct = menu.addAction("Undo Delete\tCtrl+Z");

    // Delete row only for data rows
    QAction* delAct = 0;
    if (!isAddRow(row))
        delAct = menu.addAction("Delete Row");

    QAction* chosen = menu.exec(m_table->verticalHeader()->mapToGlobal(pos));
    if (!chosen) return;

    if (undoAct && chosen == undoAct)
    {
        undoDelete();
        return;
    }
    if (delAct && chosen == delAct)
    {
        m_contextRow = row;
        deleteRowAction();
        return;
    }
}

void ViewBoolTableWidget::deleteRowAction()
{
    if (m_contextRow < 0) return;
    if (isAddRow(m_contextRow)) return;

    // Confirm
    const int ret = QMessageBox::question(
        this,
        "Delete Row",
        "Delete this row?",
        QMessageBox::Yes | QMessageBox::No,
        QMessageBox::No
    );
    if (ret != QMessageBox::Yes)
    {
        m_contextRow = -1;
        return;
    }

    // Capture for undo (single-step)
    QString left, op, right;
    if (captureRowData(m_contextRow, left, op, right))
    {
        m_undoValid = true;
        m_undoRowIndex = m_contextRow;
        m_undoLeft = left;
        m_undoOp = op;
        m_undoRight = right;
    }
    else
    {
        m_undoValid = false;
        m_undoRowIndex = -1;
    }

    // Remove
    m_table->removeRow(m_contextRow);
    m_contextRow = -1;

    // Ensure at least one add-row
    if (m_table->rowCount() == 0)
    {
        m_table->setRowCount(1);
        createRow(0, true);
    }
    else if (m_table->rowCount() == 1)
    {
        createRow(0, true);
    }
    else
    {
        createRow(addRowIndex(), true);
    }

    updateRowHeights();
    updateVerticalHeaderLabels();
}

void ViewBoolTableWidget::undoDelete()
{
    if (!m_undoValid) return;

    // Insert back before add-row
    int insertAt = m_undoRowIndex;
    if (insertAt < 0) insertAt = 0;
    if (insertAt > addRowIndex()) insertAt = addRowIndex();

    m_table->insertRow(insertAt);
    createRow(insertAt, false);
    applyRowData(insertAt, m_undoLeft, m_undoOp, m_undoRight);

    // Ensure last row remains add-row
    createRow(addRowIndex(), true);

    updateRowHeights();
    updateVerticalHeaderLabels();

    // Clear undo buffer (single-step)
    m_undoValid = false;
    m_undoRowIndex = -1;
    m_undoLeft.clear();
    m_undoOp.clear();
    m_undoRight.clear();
}

void ViewBoolTableWidget::onOk()
{
    // collect rows 0..addRowIndex()-1
    close();
}

void ViewBoolTableWidget::onCancel()
{
    emit rejected();
    close();
}
