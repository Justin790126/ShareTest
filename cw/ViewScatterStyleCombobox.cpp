#include "ViewScatterStyleCombobox.h"
#include <QPainter>

ViewScatterStyleCombobox::ViewScatterStyleCombobox(QWidget* parent)
    : QComboBox(parent)
{
    // Add all QCPScatterStyle::ScatterShape values with icons and names
    for (int i = QCPScatterStyle::ssNone+1; i <= QCPScatterStyle::ssPeace; ++i)
    {
        QCPScatterStyle::ScatterShape shape = static_cast<QCPScatterStyle::ScatterShape>(i);
        QPixmap px = scatterPixmap(shape);
        QString name = scatterShapeName(shape);
        addItem(QIcon(px), name, QVariant(i));
    }
    setIconSize(QSize(24,24));
}

QCPScatterStyle::ScatterShape ViewScatterStyleCombobox::currentShape() const
{
    return static_cast<QCPScatterStyle::ScatterShape>(itemData(currentIndex()).toInt());
}

void ViewScatterStyleCombobox::setCurrentShape(QCPScatterStyle::ScatterShape shape)
{
    for(int i = 0; i < count(); ++i)
    {
        if(itemData(i).toInt() == (int)shape)
        {
            setCurrentIndex(i);
            break;
        }
    }
}

QPixmap ViewScatterStyleCombobox::scatterPixmap(QCPScatterStyle::ScatterShape shape, int size)
{
    QPixmap pm(size, size);
    pm.fill(Qt::transparent);

    QPainter painter(&pm);
    painter.setRenderHint(QPainter::Antialiasing, true);

    QPen pen(Qt::white, 2);
    QBrush brush(Qt::white);
    QCPScatterStyle scatterStyle(shape, pen, brush, 14);
    QCPPainter* cpp = (QCPPainter*)(&painter);
    scatterStyle.drawShape(cpp, QPointF(size/2, size/2));
    return pm;
}

QString ViewScatterStyleCombobox::scatterShapeName(QCPScatterStyle::ScatterShape shape) const
{
    switch(shape)
    {
    case QCPScatterStyle::ssDot: return "Dot";
    case QCPScatterStyle::ssCross: return "Cross";
    case QCPScatterStyle::ssPlus: return "Plus";
    case QCPScatterStyle::ssCircle: return "Circle";
    case QCPScatterStyle::ssDisc: return "Disc";
    case QCPScatterStyle::ssSquare: return "Square";
    case QCPScatterStyle::ssDiamond: return "Diamond";
    case QCPScatterStyle::ssStar: return "Star";
    case QCPScatterStyle::ssTriangle: return "Triangle";
    case QCPScatterStyle::ssTriangleInverted: return "Triangle Inverted";
    case QCPScatterStyle::ssCrossSquare: return "Cross Square";
    case QCPScatterStyle::ssPlusSquare: return "Plus Square";
    case QCPScatterStyle::ssCrossCircle: return "Cross Circle";
    case QCPScatterStyle::ssPlusCircle: return "Plus Circle";
    case QCPScatterStyle::ssPeace: return "Peace";
    default: return QString("Shape %1").arg((int)shape);
    }
}