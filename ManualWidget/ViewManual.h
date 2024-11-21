#ifndef VIEW_YML_DISPLAY_H
#define VIEW_YML_DISPLAY_H

#include <QtCore>
#include <QtWidgets>

#include <QWidget>
#include <iostream>
#include <vector>

using namespace std;
#include "md2html.h"
#include "Section.h"

class ViewManual : public QWidget
{
    Q_OBJECT

public:
    explicit ViewManual(QWidget *parent = nullptr);

    QTextEdit*  teManual;

private:
    QHBoxLayout* hlytToolbar;
    QComboBox* cbbSearchBar;
    QPushButton* btnSearch;
    QHBoxLayout* hlytManualMain;
    QVBoxLayout* vlytManualTitle;
    QListWidget* lwManualTitle;
    QVBoxLayout* vlytManualContent;
    QTreeView* twManualTitle;

    QStackedWidget* stkwManualPages;

    QWidget* addManualPage(const string& btnText);
    void Widgets();
    void Layouts();


private slots:
    void handleListWidgetItemClick(QListWidgetItem *item);

};

#endif /* VIEW_YML_DISPLAY_H */