#ifndef VIEW_YML_DISPLAY_H
#define VIEW_YML_DISPLAY_H

#include <QtCore>
#include <QtWidgets>

#include <QWidget>
#include <iostream>
#include <vector>

using namespace std;


class ViewSearchResListWidgetItem : public QListWidgetItem
{
public:
    QString GetText() { return text(); }
};

class ViewManual : public QWidget
{
    Q_OBJECT

public:
    explicit ViewManual(QWidget *parent = nullptr);

    QTextEdit *teManual;

    vector<QPushButton*>* GetButtons() { return &m_vBtns; }
    vector<QTextEdit*>* GetTextEdits() { return &m_vTes; }

    QTreeWidget *GetTreeWidget() { return twTblOfContent; }
    QPushButton *GetSearchButton() { return btnSearch; }
    QListWidget *GetSearchResultList() { return lwSearchResult; }
    // get text from qcombobox
    QString GetSearchText() { return cbbSearchBar->currentText(); }
    QComboBox *GetComboBox() { return cbbSearchBar; }
    void AddManuals(const vector<QPushButton *> &buttons, const vector<QTextEdit *> &contents);

private:
    QHBoxLayout *hlytToolbar;
    QComboBox *cbbSearchBar;
    QPushButton *btnSearch;
    QHBoxLayout *hlytManualMain;
    QVBoxLayout *vlytManualTitle;
    QListWidget *lwSearchResult;
    QVBoxLayout *vlytManualContent;
    QTreeWidget *twTblOfContent;

    vector<QPushButton *> m_vBtns;
    vector<QTextEdit *> m_vTes;

    QStackedWidget *stkwManualPages;

    void Widgets();
    void Layouts();

private slots:
    void handleButtonClick();
};

#endif /* VIEW_YML_DISPLAY_H */