#ifndef VIEW_YML_DISPLAY_H
#define VIEW_YML_DISPLAY_H

#include <QtCore>
#include <QtWidgets>

#include <QWidget>

class ViewYmlDisplay : public QWidget
{
    Q_OBJECT

public:
    explicit ViewYmlDisplay(QWidget *parent = nullptr);

private:
    void paintEvent(QPaintEvent *event) override;
    QImage img1, img2, img3;
};

#endif /* VIEW_YML_DISPLAY_H */