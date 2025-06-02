#ifndef VIEWSCATTERSTYLECOMBOBOX_H
#define VIEWSCATTERSTYLECOMBOBOX_H

#include <QComboBox>
#include "qcustomplot.h"

class ViewScatterStyleCombobox : public QComboBox
{
    Q_OBJECT
public:
    explicit ViewScatterStyleCombobox(QWidget* parent = 0);

    QCPScatterStyle::ScatterShape currentShape() const;
    void setCurrentShape(QCPScatterStyle::ScatterShape shape);
    static QPixmap scatterPixmap(QCPScatterStyle::ScatterShape shape, int size = 24);

private:
    QString scatterShapeName(QCPScatterStyle::ScatterShape shape) const;
};

#endif // VIEWSCATTERSTYLECOMBOBOX_H