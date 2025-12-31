#pragma once

#include <QMainWindow>
#include <QLineEdit>
#include <QPushButton>
#include <QTreeWidget>
#include <QLabel>

#include "ModelProfTree.h"

class MainWindow : public QMainWindow
{
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = 0);
    virtual ~MainWindow();

private slots:
    void onBrowse();
    void onParse();

    void onParseStarted(const QString& path);
    void onParseFinished();
    void onParseFailed(const QString& msg);

private:
    void setupUi();
    void setupTreeColumns();

    void buildTreeWidget(); // 由 m_parser->rootStd() 建 UI
    void buildChildrenRecursive(QTreeWidgetItem* parentItem, const ProfNodeStd::Ptr& parentNode);

    void clearTree();
    void showRootMetaHeader(); // 顯示 METIS WORKER PROFILE uptime 資訊

private:
    QLineEdit*   m_pathEdit;
    QPushButton* m_browseBtn;
    QPushButton* m_parseBtn;
    QLabel*      m_metaLabel;
    QTreeWidget* m_tree;

    ModelProfTree* m_parser;
};
