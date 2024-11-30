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
    ViewSearchResListWidgetItem(const QString &text) : QListWidgetItem(text)
    {
    }
    QString GetText() { return text(); }
    void SetKeyWordPos(size_t pos) { m_sKeyWordPos = pos; }
    size_t GetKeyWordPos() { return m_sKeyWordPos; }
    void SetSearchText(string text) { m_sSearchText = text; }
    int GetBtnIdx() { return m_iBtnIdx; }

    string GetSearchText() { return m_sSearchText; }
    void SetBtnIdx(int idx) { m_iBtnIdx = idx; }

private:
    size_t m_sKeyWordPos = 0;
    string m_sSearchText;
    int m_iBtnIdx = 0;
};

class ViewManual : public QWidget
{
    Q_OBJECT

public:
    explicit ViewManual(QWidget *parent = nullptr);

    QTextEdit *teManual;

    vector<QPushButton *> *GetButtons() { return &m_vBtns; }
    vector<QTextEdit *> *GetTextEdits() { return &m_vTes; }

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