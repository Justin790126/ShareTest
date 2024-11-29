#ifndef VIEW_YML_DISPLAY_H
#define VIEW_YML_DISPLAY_H

#include <QtCore>
#include <QtWidgets>

#include <QWidget>
#include <iostream>
#include <vector>

using namespace std;

#include "Section.h"

class ViewManual : public QWidget
{
    Q_OBJECT

public:
    explicit ViewManual(QWidget *parent = nullptr);

    QTextEdit*  teManual;

    QTreeWidget* GetTreeWidget() { return twTblOfContent; }

    void AddManuals(const vector<QPushButton*>& buttons, const vector<QTextEdit*>& contents);

private:
    QHBoxLayout* hlytToolbar;
    QComboBox* cbbSearchBar;
    QPushButton* btnSearch;
    QHBoxLayout* hlytManualMain;
    QVBoxLayout* vlytManualTitle;
    QListWidget* lwSearchResult;
    QVBoxLayout* vlytManualContent;
    QTreeWidget* twTblOfContent;

    vector<QPushButton*> m_vBtns;
    vector<QTextEdit*> m_vTes;

    QStackedWidget* stkwManualPages;

    void Widgets();
    void Layouts();


private slots:
    void handleButtonClick();

};

#endif /* VIEW_YML_DISPLAY_H */