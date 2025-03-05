#include "ViewTensorBoard.h"

ViewTensorBoard::ViewTensorBoard(QWidget *parent)
    : QWidget(parent)
{
    UI();
}

void ViewTensorBoard::widgets()
{
    tbwMain = new QTabWidget(this);
    QLabel* lblTitle = new QLabel("TensorBoard");
    // set font size
    lblTitle->setFont(QFont("Arial", 18, QFont::Bold));
    tbwMain->setCornerWidget(lblTitle, Qt::TopLeftCorner);

    viewTimeSeries = new ViewTimeSeries(this);

    tbwMain->addTab(viewTimeSeries, "TIME SERIES");
    // tbwMain->addTab(new QWidget(), tr("Runs"));
    // tbwMain->addTab(new QWidget(), tr("Events"));


}

vector<QTreeWidgetItem*> ViewTensorBoard::CreateJobItems(const vector<string>& jobs)
{
    QTreeWidget* tw = viewTimeSeries->GetTwJobsTree();
    vector<QTreeWidgetItem*> items;
    items.reserve(jobs.size());
    int idx = 0;
    // iterate jobs and create QTreeWidgetItem in tw
    for (const auto& job : jobs)
    {
        ViewTfTreeItem* item = new ViewTfTreeItem(tw);
        item->setText(0, QString::fromStdString(job));
        // set item checkable
        
        item->setCheckState(0, Qt::Unchecked);
        item->SetTfLiveIdx(idx);
        items.emplace_back(item);

        idx++;
    }
    return items;
}

QWidget* ViewTensorBoard::CreateChartSection(QString title, ChartInfo* chartInfo)
{
    return viewTimeSeries->AddChartSection(title, chartInfo);
}

void ViewTensorBoard::layouts()
{
    QVBoxLayout *vlytMain = new QVBoxLayout(this);
    {
        vlytMain->addWidget(tbwMain);
    }
    this->setLayout(vlytMain);
}

void ViewTensorBoard::UI()
{
    widgets();
    layouts();

    // Set up the window
    setWindowTitle(tr("Iht TensorBoard"));
    setMinimumSize(800, 600);
    
}