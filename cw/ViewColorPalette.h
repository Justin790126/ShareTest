#ifndef COLORPALETTE_H
#define COLORPALETTE_H

#include <QWidget>
#include <QColor>
#include <QVector>

class ColorPalette : public QWidget
{
    Q_OBJECT
public:
    explicit ColorPalette(QWidget *parent = nullptr);

    QSize sizeHint() const override;
    QColor selectedColor() const;

signals:
    void colorSelected(const QColor &color);

protected:
    void paintEvent(QPaintEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;

private:
    QVector<QVector<QColor>> m_colors; // 2D grid of colors
    int m_cellSize;
    int m_cellSpacing;
    int m_rows, m_cols;
    QColor m_selectedColor;
    QRect cellRect(int row, int col) const;
    void populateColors();
};

#endif // COLORPALETTE_H