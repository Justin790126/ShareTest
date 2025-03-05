#ifndef VIEW_TENSORBOARD
#define VIEW_TENSORBOARD

#include <iostream>
#include <string>
#include <vector>
#include <QtGui>
#include "ViewTimeSeries.h"

using namespace std;

class ViewTfTreeItem : public QTreeWidgetItem
{
public:
    // Constructor with optional parent QTreeWidget
    explicit ViewTfTreeItem(QTreeWidget* parent = 0)
        : QTreeWidgetItem(parent)
    {
        // Initialization code can go here if needed
    }

    // Constructor with parent and text for a single column
    ViewTfTreeItem(QTreeWidget* parent, const QString& text)
        : QTreeWidgetItem(parent)
    {
        setText(0, text); // Set text in the first column
    }

    // Destructor
    ~ViewTfTreeItem()
    {
        // Cleanup code can go here if needed
        // QTreeWidgetItem handles its own memory management for basic cases
    }
    int GetTfLiveIdx() const { return m_iTfLiveIdx; }
    void SetTfLiveIdx(int idx) { m_iTfLiveIdx = idx; }
private:
    int m_iTfLiveIdx;
};

class ViewTensorBoard : public QWidget
{
    Q_OBJECT
public:
    ViewTensorBoard(QWidget *parent = NULL);
    ~ViewTensorBoard() = default;
    
    vector<QTreeWidgetItem*> CreateJobItems(const vector<string>& jobs);

    QTreeWidget* GetTwJobsTree() { return viewTimeSeries->GetTwJobsTree(); }

    QWidget* CreateChartSection(QString title, ChartInfo* chartInfo);

private:
    void widgets();
    void layouts();
    void UI();

    QTabWidget* tbwMain;

    ViewTimeSeries* viewTimeSeries;
};

#endif /* VIEW_TENSORBOARD */