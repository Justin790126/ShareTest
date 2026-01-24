#ifndef VIEWBOOLTABLEWIDGET_H
#define VIEWBOOLTABLEWIDGET_H

#include <QWidget>
#include <QString>

class QTableWidget;

class ViewBoolTableWidget : public QWidget
{
    Q_OBJECT
public:
    explicit ViewBoolTableWidget(QWidget* parent = 0);

signals:
    void rejected();

private slots:
    void onOk();
    void onCancel();

    // double-click '+' to add
    void onVHeaderDoubleClicked(int logicalIndex);

    // vertical header context menu
    void onVHeaderContextMenu(const QPoint& pos);
    void deleteRowAction();
    void undoDelete();

private:
    void initTable();

    void createRow(int row, bool isAddRow);
    int addRowIndex() const;
    bool isAddRow(int row) const;

    void appendDataRow();              // insert before add-row
    void updateVerticalHeaderLabels();
    void updateRowHeights();

    bool captureRowData(int row, QString& left, QString& op, QString& right) const;
    void applyRowData(int row, const QString& left, const QString& op, const QString& right);

private:
    QTableWidget* m_table;
    int m_contextRow;

    // undo buffer (single-step)
    bool   m_undoValid;
    int    m_undoRowIndex;
    QString m_undoLeft, m_undoOp, m_undoRight;
};

#endif
