#ifndef COLORCOMBOBOX_H
#define COLORCOMBOBOX_H

#include <QComboBox>
#include "ViewColorPalette.h"

class ViewColorCombobox : public QComboBox
{
    Q_OBJECT
public:
    explicit ViewColorCombobox(QWidget *parent = nullptr);

    QColor currentColor() const;

signals:
    void colorChanged(const QColor &color);

protected:
    void showPopup() override;

private slots:
    void onColorSelected(const QColor &color);

private:
    ColorPalette *m_palette;
    void updateIcon(const QColor &color);
};

#endif // COLORCOMBOBOX_H