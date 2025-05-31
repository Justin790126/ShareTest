#include "ViewColorPalette.h"
#include <QPainter>
#include <QMouseEvent>

ColorPalette::ColorPalette(QWidget *parent)
    : QWidget(parent), m_cellSize(24), m_cellSpacing(4), m_rows(6), m_cols(8)
{
    populateColors();
    setWindowFlags(Qt::Popup);
    setMouseTracking(true);
    setFixedSize(m_cols * (m_cellSize + m_cellSpacing) + m_cellSpacing,
                 m_rows * (m_cellSize + m_cellSpacing) + m_cellSpacing);
}

void ColorPalette::populateColors()
{
    // For demonstration, fill with a simple color pattern (change as needed)
    m_colors.resize(m_rows);
    for (int row = 0; row < m_rows; ++row) {
        m_colors[row].resize(m_cols);
        for (int col = 0; col < m_cols; ++col) {
            int hue = (col * 360) / m_cols;
            int sat = 255;
            int val = 255 - (row * 40);
            m_colors[row][col] = QColor::fromHsv(hue, sat, val);
        }
    }
}

QSize ColorPalette::sizeHint() const
{
    return QSize(m_cols * (m_cellSize + m_cellSpacing) + m_cellSpacing,
                 m_rows * (m_cellSize + m_cellSpacing) + m_cellSpacing);
}

QColor ColorPalette::selectedColor() const
{
    return m_selectedColor;
}

QRect ColorPalette::cellRect(int row, int col) const
{
    int x = m_cellSpacing + col * (m_cellSize + m_cellSpacing);
    int y = m_cellSpacing + row * (m_cellSize + m_cellSpacing);
    return QRect(x, y, m_cellSize, m_cellSize);
}

void ColorPalette::paintEvent(QPaintEvent *)
{
    QPainter p(this);
    for (int row = 0; row < m_rows; ++row) {
        for (int col = 0; col < m_cols; ++col) {
            QRect rect = cellRect(row, col);
            QColor color = m_colors[row][col];
            p.setPen(Qt::NoPen);
            p.setBrush(color);
            p.drawRect(rect);

            // Draw border if selected
            if (color == m_selectedColor) {
                p.setPen(QPen(Qt::black, 2));
                p.setBrush(Qt::NoBrush);
                p.drawRect(rect.adjusted(1, 1, -1, -1));
            }
        }
    }
}

void ColorPalette::mousePressEvent(QMouseEvent *event)
{
    for (int row = 0; row < m_rows; ++row) {
        for (int col = 0; col < m_cols; ++col) {
            if (cellRect(row, col).contains(event->pos())) {
                m_selectedColor = m_colors[row][col];
                emit colorSelected(m_selectedColor);
                update();
                close(); // close popup after selection
                return;
            }
        }
    }
}