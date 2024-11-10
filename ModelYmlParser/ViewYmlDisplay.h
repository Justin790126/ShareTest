#ifndef VIEW_YML_DISPLAY_H
#define VIEW_YML_DISPLAY_H

#include <QtCore>
#include <QtWidgets>

#include <QWidget>

#include "md2html.h"

class ViewYmlDisplay : public QWidget
{
    Q_OBJECT

public:
    explicit ViewYmlDisplay(QWidget *parent = nullptr);

    QTreeWidget* twYmlDisplay;
    QSplitter* spltMain;
    QTextEdit*  teManual;

private:
    void Widgets();
    void Layouts();

};

#endif /* VIEW_YML_DISPLAY_H */