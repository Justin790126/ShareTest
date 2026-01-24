#ifndef VIEWOPERATIONWIDGET_H
#define VIEWOPERATIONWIDGET_H

#include <QWidget>
#include <QString>

class QComboBox;
class QLabel;

class ViewOperationWidget : public QWidget
{
    Q_OBJECT
public:
    explicit ViewOperationWidget(QWidget* parent = 0);

    QString operation() const;
    void setOperation(const QString& op);

signals:
    void operationChanged(const QString& op);

private slots:
    void onComboChanged(const QString& op);

private:
    QPixmap makeOpPixmap(const QString& op) const;
    void refreshPixmap();

private:
    QComboBox* m_combo;
    QLabel*    m_preview;
};

#endif
