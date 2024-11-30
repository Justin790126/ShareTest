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
        if (level == 0) {
            nodes[level] = new QTreeWidgetItem(tw);
            nodes[level]->setText(0, QString::fromStdString(""));
            nodes[level]->setText(1, QString::fromStdString(node->GetUrl()));
            tw->addTopLevelItem(nodes[level]);
        } else if(level < nodes.size()+1) {
            nodes[level] = new QTreeWidgetItem();
            nodes[level]->setText(0, QString::fromStdString(""));
            nodes[level]->setText(1, QString::fromStdString(node->GetUrl()));
            // add current node to parent node
            nodes[level - 1]->addChild(nodes[level]);

            QPushButton* btn = new QPushButton(QString::fromStdString(node->GetKey()));
            tw->setItemWidget(nodes[level], 0, btn);
            btns.push_back(btn);

            QTextEdit* te = new QTextEdit(QString::fromStdString(node->GetHtmlContent()));
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
    void handleButtonClick()
    {
        if (!model) return;
        model->Search("123");
        vector<SearchInfo> *infos = model->GetSearchInfos();
        QListWidget *lw = view->GetSearchResultList();
        lw->clear();
        for (SearchInfo info : *infos) {
            lw->addItem(QString::fromStdString(info.GetInfo().c_str()));
        }
        lw->setVisible(true);
    }
};
