#ifndef VIEWCOLEDIT_H
#define VIEWCOLEDIT_H

#include <QScrollArea>
#include <QApplication>
#include <QLabel>
#include <QVBoxLayout>
#include <QListWidget>
#include <QPushButton>
#include <QStyle>
#include <QIcon>

class ViewColEdit : public QScrollArea
{
    Q_OBJECT

public:
    ViewColEdit(QWidget *parent = NULL);

    void Widgets();
    void Layout();

    QListWidget* lwAllCols;
    QListWidget* lwDesireCols;

    QPushButton* btnArrLeft;
    QPushButton* btnArrRight;
    QHBoxLayout *lytMain;
    QWidget     *widMain;

    // Add any additional functionality or custom methods as needed
};

#endif // VIEWCOLEDIT_H
