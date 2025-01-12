#include "lcDoc.h"

lcDoc::lcDoc(QWidget *parent) : QWidget(parent)
{
    // Model operation
    model = new ModelMdReader;
    string ipt = "./doc/README.md";
    model->SetFname(ipt);
    model->SetRootPath("/home/justin126/workspace/ShareTest/lcDoc/");
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
                                //   cout << *node << endl;

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

                                    //   string htmlPath = node->GetHtmlPath();
                                    //   string htmlContent = node->GetHtmlContent();
                                    //   // write html content
                                    //   std::ofstream outfile(htmlPath);
                                    //   outfile << htmlContent;
                                    //   outfile.close();

                                      QTextBrowser *te = new QTextBrowser();
                                      te->setHtml(QString::fromStdString(node->GetHtmlContent()));
                                      te->setReadOnly(true);
                                      tes.push_back(te);
                                  } });

    tw->expandAll();
    tw->resizeColumnToContents(1);

    view->AddManuals(btns, tes);

    QVBoxLayout *vl = new QVBoxLayout;
    vl->addWidget(view);
    this->setLayout(vl);
    // this->showMaximized();

    // controller operation
    connect(view->GetSearchButton(), SIGNAL(clicked()), this, SLOT(handleButtonClick()));
    // connect qcombobox text change event to trigger search function
    connect(view->GetComboBox(), SIGNAL(editTextChanged(const QString &)), this, SLOT(handleSearchTextChanged(const QString &)));
    connect(view->GetSearchResultList(), SIGNAL(itemClicked(QListWidgetItem *)),
            this, SLOT(handleSearchResultClicked(QListWidgetItem *)));
}

lcDoc::~lcDoc()
{
}

void lcDoc::closeEvent(QCloseEvent *event)
{
    // handle close event
}

void lcDoc::handleSearchTextChanged(const QString &msg)
{
    if (msg.isEmpty())
    {
        QListWidget *lw = view->GetSearchResultList();
        lw->clear();
        lw->setVisible(false);
    }
}
void lcDoc::handleButtonClick()
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
        ViewSearchResListWidgetItem *item = new ViewSearchResListWidgetItem(QString::fromStdString(info.GetInfo().c_str()));
        item->SetKeyWordPos(info.GetKeyPos());
        item->SetBtnIdx(info.GetBtnIdx());
        item->SetSearchText(text);
        lw->addItem(item);
    }
    lw->setVisible(true);
}

void lcDoc::handleSearchResultClicked(QListWidgetItem *item)
{
    string selText = item->text().toStdString();
    ViewSearchResListWidgetItem *it = dynamic_cast<ViewSearchResListWidgetItem *>(item);
    SearchInfo info;
    // string resultText = info.GetSearchResultFromInfo(selText);
    int btnIdx = it->GetBtnIdx();
    vector<QPushButton *> *btns = view->GetButtons();
    if (btnIdx >= 0 && btnIdx < btns->size())
    {
        QPushButton *btn = btns->at(btnIdx);
        btn->click();
    }

    // highlight text in text edit
    QTextEdit *te = view->GetTextEdits()->at(btnIdx);
    te->setFocus();
    string searchText = it->GetSearchText();
    size_t pos = te->toPlainText().indexOf(searchText.c_str());
    if (pos)
    {
        QTextCursor cursor = te->textCursor();
        int startPosition = pos;
        cursor.setPosition(startPosition);
        cursor.movePosition(QTextCursor::MoveOperation::Right, QTextCursor::MoveMode::KeepAnchor, searchText.size());
        te->setTextCursor(cursor);
        QScrollBar *scrollBar = te->verticalScrollBar();
        QRect cursorRect = te->cursorRect(cursor);
        int contentHeight = te->document()->size().height();
        int viewportHeight = te->viewport()->height();

        int scrollValue = qMax(0, qMin(cursorRect.top() - (viewportHeight / 2),
                                       contentHeight - viewportHeight));

        scrollBar->setValue(scrollValue);
        te->ensureCursorVisible();
    }
}