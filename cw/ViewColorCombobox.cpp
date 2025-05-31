#include "ViewColorCombobox.h"
#include <QPainter>
#include <QApplication>
#include <QScreen>
#include <QDesktopWidget>

ViewColorCombobox::ViewColorCombobox(QWidget *parent)
    : QComboBox(parent), m_palette(nullptr)
{
    setEditable(false);
    addItem(""); // slot for the color icon
    setMinimumWidth(60);
    updateIcon(Qt::black); // default color
}

QColor ViewColorCombobox::currentColor() const
{
    QVariant var = itemData(0, Qt::UserRole);
    return var.isValid() ? var.value<QColor>() : Qt::black;
}

void ViewColorCombobox::showPopup()
{
    if (!m_palette) {
        m_palette = new ColorPalette();
        connect(m_palette, SIGNAL(colorSelected), this, SLOT(onColorSelected));
    }
    QPoint below = mapToGlobal(QPoint(0, height()));
    // Ensure popup fits on screen
    // FIX this
    QRect screenRect = QApplication::desktop()->screenGeometry(this);
    QSize paletteSize = m_palette->sizeHint();
    if (below.y() + paletteSize.height() > screenRect.bottom())
        below.setY(mapToGlobal(QPoint(0, 0)).y() - paletteSize.height());
    m_palette->move(below);
    m_palette->show();
    m_palette->raise();
}

void ViewColorCombobox::onColorSelected(const QColor &color)
{
    updateIcon(color);
    setItemData(0, color, Qt::UserRole);
    emit colorChanged(color);
}

void ViewColorCombobox::updateIcon(const QColor &color)
{
    QPixmap pix(32, 16);
    pix.fill(Qt::transparent);
    QPainter p(&pix);
    p.setBrush(color);
    p.setPen(Qt::gray);
    p.drawRect(0, 0, pix.width() - 1, pix.height() - 1);
    p.end();
    setItemIcon(0, QIcon(pix));
    setCurrentIndex(0);
}