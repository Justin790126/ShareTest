#include "MainWindow.h"

#include <QWidget>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QFileDialog>
#include <QStatusBar>
#include <QStringList>

#include "ViewProfTreeItem.h"

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
    , m_pathEdit(0)
    , m_browseBtn(0)
    , m_parseBtn(0)
    , m_metaLabel(0)
    , m_tree(0)
    , m_parser(0)
{
    setupUi();
    setupTreeColumns();

    m_parser = new ModelProfTree(this);

    connect(m_browseBtn, SIGNAL(clicked()), this, SLOT(onBrowse()));
    connect(m_parseBtn, SIGNAL(clicked()), this, SLOT(onParse()));

    connect(m_parser, SIGNAL(parseStarted(QString)), this, SLOT(onParseStarted(QString)));
    connect(m_parser, SIGNAL(parseFinished()), this, SLOT(onParseFinished()));
    connect(m_parser, SIGNAL(parseFailed(QString)), this, SLOT(onParseFailed(QString)));

    statusBar()->showMessage("Ready");
}

MainWindow::~MainWindow()
{
    if (m_parser) {
        m_parser->requestStop();
        m_parser->wait();
    }
}

void MainWindow::setupUi()
{
    QWidget* c = new QWidget(this);
    setCentralWidget(c);

    m_pathEdit = new QLineEdit(c);
    m_pathEdit->setPlaceholderText("Select profile log file...");

    m_browseBtn = new QPushButton("Browse", c);
    m_parseBtn  = new QPushButton("Parse", c);

    m_metaLabel = new QLabel(c);
    m_metaLabel->setText("Meta: (empty)");
    m_metaLabel->setWordWrap(true);

    m_tree = new QTreeWidget(c);

    QHBoxLayout* top = new QHBoxLayout();
    top->addWidget(m_pathEdit, 1);
    top->addWidget(m_browseBtn);
    top->addWidget(m_parseBtn);

    QVBoxLayout* root = new QVBoxLayout();
    root->addLayout(top);
    root->addWidget(m_metaLabel);
    root->addWidget(m_tree, 1);

    c->setLayout(root);

    setWindowTitle("Profile Tree Viewer (Qt4.8)");
    resize(1100, 700);
}

void MainWindow::setupTreeColumns()
{
    QStringList headers;
    headers << "Name"
            << "Total(ms)"
            << "CPU(ms)"
            << "Calls"
            << "Percent(%)"
            << "Child(%)"
            << "Mem(MB)";
    m_tree->setColumnCount(ViewProfTreeItem::COL_COUNT);
    m_tree->setHeaderLabels(headers);
    m_tree->setUniformRowHeights(true);
    m_tree->setAllColumnsShowFocus(true);
}

void MainWindow::onBrowse()
{
    QString path = QFileDialog::getOpenFileName(this, "Select Profile Log");
    if (!path.isEmpty())
        m_pathEdit->setText(path);
}

void MainWindow::onParse()
{
    if (!m_parser) return;

    // 若前一次還在跑，先中止（你也可以改成 disable 按鈕）
    if (m_parser->isRunning()) {
        m_parser->requestStop();
        m_parser->wait();
    }

    clearTree();
    m_metaLabel->setText("Meta: (parsing...)");

    QString path = m_pathEdit->text().trimmed();
    if (path.isEmpty()) {
        statusBar()->showMessage("Please select a file.");
        return;
    }

    m_parser->setFilePath(path);
    m_parser->start();
}

void MainWindow::onParseStarted(const QString& path)
{
    statusBar()->showMessage(QString("Parsing: %1").arg(path));
    m_parseBtn->setEnabled(false);
}

void MainWindow::onParseFinished()
{
    statusBar()->showMessage("Parse finished.");
    m_parseBtn->setEnabled(true);

    buildTreeWidget();
}

void MainWindow::onParseFailed(const QString& msg)
{
    statusBar()->showMessage(QString("Parse failed: %1").arg(msg));
    m_parseBtn->setEnabled(true);

    m_metaLabel->setText(QString("Meta: (failed) %1").arg(msg));
}

void MainWindow::clearTree()
{
    m_tree->clear();
}

static QString toQ(const std::string& s)
{
    return QString::fromLocal8Bit(s.c_str());
}

void MainWindow::showRootMetaHeader()
{
    std::shared_ptr<const ProfNodeStd> root = m_parser->rootStd();
    if (!root) {
        m_metaLabel->setText("Meta: (null root)");
        return;
    }

    const RootMetaStd& meta = root->meta();

    QString text;
    if (!meta.profileTitle().empty()) {
        text += QString("PROFILE: %1\n").arg(toQ(meta.profileTitle()));
    }
    if (!meta.totalUptime().empty()) {
        text += QString("Total uptime : %1\n").arg(toQ(meta.totalUptime()));
    }
    if (!meta.clientUptime().empty()) {
        text += QString("Client uptime: %1 (%2)\n")
                    .arg(toQ(meta.clientUptime()))
                    .arg(toQ(meta.clientPercent()));
    }
    if (!meta.serverUptime().empty()) {
        text += QString("Server uptime: %1 (%2)\n")
                    .arg(toQ(meta.serverUptime()))
                    .arg(toQ(meta.serverPercent()));
    }

    if (text.isEmpty())
        text = "Meta: (empty)";
    m_metaLabel->setText(text.trimmed());
}

void MainWindow::buildTreeWidget()
{
    clearTree();
    showRootMetaHeader();

    std::shared_ptr<const ProfNodeStd> rootConst = m_parser->rootStd();
    // rootStd() 回傳 const；但 children 的節點本身是 shared_ptr<ProfNodeStd>（Ptr）
    // 這裡我們只讀，所以把 rootConst 轉成非 const 指標是不需要的。
    // 但 children() 回的是 ProfNodeStd::Ptr vector（非 const class 的 Ptr）。
    // 因為 rootConst 是 const，所以我們直接用 rootConst 取 children() 會回 const vector<Ptr>&，仍可讀。

    if (!rootConst) return;

    const std::vector<ProfNodeStd::Ptr>& top = rootConst->children();
    for (size_t i = 0; i < top.size(); ++i) {
        ProfNodeStd::Ptr n = top[i];
        ViewProfTreeItem* item = new ViewProfTreeItem(n);
        m_tree->addTopLevelItem(item);
        buildChildrenRecursive(item, n);
    }

    m_tree->expandToDepth(1);
}

void MainWindow::buildChildrenRecursive(QTreeWidgetItem* parentItem, const ProfNodeStd::Ptr& parentNode)
{
    const std::vector<ProfNodeStd::Ptr>& kids = parentNode->children();
    for (size_t i = 0; i < kids.size(); ++i) {
        ProfNodeStd::Ptr c = kids[i];
        ViewProfTreeItem* childItem = new ViewProfTreeItem(c);
        parentItem->addChild(childItem);
        buildChildrenRecursive(childItem, c);
    }
}
