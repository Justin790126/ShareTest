#include <QApplication>
#include <QWidget>
#include "ModelMdReader.h"
#include "ViewManual.h"

class MainWindow : public QWidget
{
    Q_OBJECT
public:
    ModelMdReader *model;
    ViewManual *view;

    MainWindow(QWidget *parent = nullptr) : QWidget(parent)
    {
        // Model operation
        model = new ModelMdReader;
        string ipt = "/Users/justinchang/workspace/ShareTest/ManualWidget/doc/README.md";
        model->SetFname(ipt);
        model->testRun();

        MdNode *m_sRoot = model->GetRoot();
        int maxLevel = model->GetMaxLevel();
        view = new ViewManual;

        // View operation
        QTreeWidget *tw = view->GetTreeWidget();
        cout << "Max Level: " << maxLevel << endl;
        vector<QTreeWidgetItem *> nodes(maxLevel + 1, NULL);
        vector<QPushButton *> btns;
        btns.reserve(maxLevel + 1);
        vector<QTextEdit *> tes;
        tes.reserve(maxLevel + 1);
        model->TraverseMdNode(m_sRoot, [&tw, &nodes, &btns, &tes](MdNode *node)
                              {
                                  cout << *node << endl;

                                  int level = node->GetLevel();
                                  if (level == 0)
                                  {
                                      nodes[level] = new QTreeWidgetItem(tw);
                                      nodes[level]->setText(0, QString::fromStdString(""));
                                      nodes[level]->setText(1, QString::fromStdString(node->GetUrl()));
                                      tw->addTopLevelItem(nodes[level]);
                                  }
                                  else if (level < nodes.size() + 1)
                                  {
                                      nodes[level] = new QTreeWidgetItem();
                                      nodes[level]->setText(0, QString::fromStdString(""));
                                      nodes[level]->setText(1, QString::fromStdString(node->GetUrl()));
                                      // add current node to parent node
                                      nodes[level - 1]->addChild(nodes[level]);
                                  }

                                  if (nodes[level])
                                  {
                                      QPushButton *btn = new QPushButton(QString::fromStdString(node->GetKey()));
                                      tw->setItemWidget(nodes[level], 0, btn);
                                      btns.push_back(btn);

                                      QTextEdit *te = new QTextEdit(QString::fromStdString(node->GetHtmlContent()));
                                      te->setReadOnly(true);
                                      tes.push_back(te);
                                  } });

        tw->expandAll();
        tw->resizeColumnToContents(1);

        view->AddManuals(btns, tes);

        QVBoxLayout *vl = new QVBoxLayout;
        vl->addWidget(view);
        this->setLayout(vl);
        this->showMaximized();

        // controller operation
        connect(view->GetSearchButton(), SIGNAL(clicked()), this, SLOT(handleButtonClick()));
        // connect qcombobox text change event to trigger search function
        connect(view->GetComboBox(), SIGNAL(editTextChanged(const QString &)), this, SLOT(handleSearchTextChanged(const QString &)));
        connect(view->GetSearchResultList(), SIGNAL(itemClicked(QListWidgetItem *)),
                this, SLOT(handleSearchResultClicked(QListWidgetItem *)));
    }
    ~MainWindow()
    {
        // cleanup
    }
    void closeEvent(QCloseEvent *event) override
    {
        // handle close event
    }
    // other UI elements and methods
private slots:
    void handleSearchTextChanged(const QString &msg)
    {
        if (msg.isEmpty())
        {
            QListWidget *lw = view->GetSearchResultList();
            lw->clear();
            lw->setVisible(false);
        }
    }
    void handleButtonClick()
    {
        if (!model)
            return;
        string text = view->GetSearchText().toStdString();
        if (text.empty())
            return;
        model->Search(text);
        vector<SearchInfo> *infos = model->GetSearchInfos();
        QListWidget *lw = view->GetSearchResultList();
        lw->clear();
        for (SearchInfo info : *infos)
        {
            QListWidgetItem *item = new QListWidgetItem(QString::fromStdString(info.GetInfo().c_str()));
            item->setData(Qt::UserRole, QVariant::fromValue(info.GetBtnIdx()));
            lw->addItem(item);
        }
        lw->setVisible(true);
    }

    void handleSearchResultClicked(QListWidgetItem *item)
    {
        string selText = item->text().toStdString();
        SearchInfo info;
        // string resultText = info.GetSearchResultFromInfo(selText);
        int btnIdx = item->data(Qt::UserRole).toInt();
        vector<QPushButton *> *btns = view->GetButtons();
        if (btnIdx >= 0 && btnIdx < btns->size())
        {
            QPushButton *btn = btns->at(btnIdx);
            btn->click();
        }

        // highlight text in text edit
        QTextEdit *te = view->GetTextEdits()->at(btnIdx);
        te->setFocus();
        QString htmlContent = te->toHtml();

        // FIXME: find search key, and find position in html content
        // and set cursor to the position
        // TODO: optimize this part by using QTextDocument::find() or QTextDocument::findText() method
        // int pos = htmlContent.indexOf(QString::fromStdString(selText));
        // if (pos >= 0)
        // {
        //     cout << "pos: " << pos << endl;
        //     QTextCursor c = te->textCursor();
            
        //     c.setPosition(pos);
            
        //     c.setPosition(pos + selText.length(), QTextCursor::KeepAnchor);
        //     // c.movePosition(QTextCursor::Start);
        //     // c.movePosition(QTextCursor::WordRight, QTextCursor::KeepAnchor);
        //     // Set the cursor back to the textEdit and ensure it's visible
        //     te->setTextCursor(c);
        //     te->ensureCursorVisible();
        // }
    }
};
