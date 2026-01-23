#ifndef BOOLTABLEWIDGET_H
#define BOOLTABLEWIDGET_H

#include <QWidget>
#include <QString>

class QTableWidget;
class QLabel;
class QComboBox;

class BoolTableWidget : public QWidget
{
    Q_OBJECT
public:
    explicit BoolTableWidget(QWidget* parent = 0);

signals:
    void accepted(const QString& leftPath,
                  const QString& op,
                  const QString& rightPath);
    void rejected();

private slots:
    void openFileLeft();
    void openFileRight();
    void operationChanged(const QString& op);
    void onOk();
    void onCancel();

private:
    QPixmap makeOpPixmap(const QString& op) const;
    void refreshOpPixmap();

private:
    QTableWidget* m_table;

    QLabel*    m_leftPathLabel;
    QLabel*    m_rightPathLabel;

    // merged operation cell
    QComboBox* m_combo;
    QLabel*    m_opImageLabel;

    QString m_leftPath;
    QString m_rightPath;
};

#endif
