#include "ViewManual.h"
#include <QVector>

ViewManual::ViewManual(QWidget *parent)
    : QWidget(parent)
{

    Widgets();
    Layouts();
}


void ViewManual::Widgets()
{
    stkwManualPages = new QStackedWidget;
    lwSearchResult = new QListWidget;
    lwSearchResult->setVisible(false);
    twTblOfContent = new QTreeWidget;
    twTblOfContent->setColumnCount(2);
    QStringList header;
    header << "Subject" << "Summay";
    twTblOfContent->setHeaderLabels(header);

    cbbSearchBar = new QComboBox;
    cbbSearchBar->setEditable(true);
    btnSearch = new QPushButton("Search");
}

void ViewManual::AddManuals(const vector<QPushButton*>& buttons, const vector<QTextEdit*>& contents)
{
    m_vBtns = buttons;
    m_vTes = contents;
    for (size_t i = 0; i < buttons.size(); i++)
    {
        stkwManualPages->addWidget(contents[i]);
        connect(buttons[i], SIGNAL(clicked()), this, SLOT(handleButtonClick()));
    }
    
}

void ViewManual::handleButtonClick()
{
    int index = m_vBtns.size();
    for (size_t i = 0; i < m_vBtns.size(); i++)
    {
        if (m_vBtns[i] == sender())
        {
            index = i;
            break;
        }
    }
    stkwManualPages->setCurrentIndex(index);
}

void ViewManual::Layouts()
{
    QVBoxLayout* vlytMain = new QVBoxLayout;
    vlytMain->setContentsMargins(0,0,0,0);
    {
        hlytToolbar = new QHBoxLayout;
        {
            hlytToolbar->addWidget(cbbSearchBar,3);
            hlytToolbar->addWidget(btnSearch,1);
            hlytToolbar->addStretch(6);
        }

        hlytManualMain = new QHBoxLayout;
        hlytManualMain->setContentsMargins(0,0,0,0);
        {  
            
            QSplitter *splitter = new QSplitter;
            QWidget* widExplorer = new QWidget;
            {
                
                QVBoxLayout *vlytWidExplorer = new QVBoxLayout(widExplorer);
                vlytWidExplorer->setContentsMargins(0,0,0,0);
                vlytWidExplorer->addWidget(twTblOfContent);
                vlytWidExplorer->addWidget(lwSearchResult);
            }
            splitter->addWidget(widExplorer);
            splitter->addWidget(stkwManualPages);
            hlytManualMain->addWidget(splitter);
        }
        vlytMain->addLayout(hlytToolbar);
        vlytMain->addLayout(hlytManualMain);
    }
    setLayout(vlytMain);
}
