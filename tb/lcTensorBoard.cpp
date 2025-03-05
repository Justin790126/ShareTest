#include "lcTensorBoard.h"

LcTensorBoard::LcTensorBoard(TbArgs args, QObject *parent)
    : QObject(parent)
{
    tbArgs = args;

    if (!view) {
        view = new ViewTensorBoard();
        view->show();
    }
    // feed logdir to ModelTfWatcher
    fsWatcher = new ModelTfWatcher;
    if (!tbArgs.m_sLogDir.empty()) {
        fsWatcher->SetLogDir(tbArgs.m_sLogDir);
        disconnect(fsWatcher, SIGNAL(tfFileChanged()), this, SLOT(handleTfFileChanged()));
        connect(fsWatcher, SIGNAL(tfFileChanged()), this, SLOT(handleTfFileChanged()));
        fsWatcher->start();
    }
    

}

LcTensorBoard::~LcTensorBoard()
{
    if (fsWatcher) {
        fsWatcher->SetWatcher(false);
    }
    if (view) delete view;
    view = NULL;
}

void LcTensorBoard::handleTreeItemClicked(QTreeWidgetItem* item, int idx)
{
    QTreeWidget* tw = view->GetTwJobsTree();
    ViewTfTreeItem* it = (ViewTfTreeItem*) item;
    if (!it) return;
    int liveIdx = it->GetTfLiveIdx();

    vector<TfLiveInfo*>* infos = fsWatcher->GetLiveInfo();
    if (!infos) return;
    if (infos->empty()) return;
    TfLiveInfo* info = (*infos)[liveIdx];
    if (!info) return;

    cout << item->text(0).toStdString() << endl;
    // update info
    ViewLineChartProps* vpAcc = (ViewLineChartProps*)m_vChartSections[0];
    ViewLineChartProps* vpLoss = (ViewLineChartProps*)m_vChartSections[1];

    // check item is checked
    bool visibility = (item->checkState(0) == Qt::Checked);
    vpAcc->SetLineChartVisibility(liveIdx, visibility);
    vpLoss->SetLineChartVisibility(liveIdx, visibility);

}


void LcTensorBoard::handleTfFileChanged()
{
    cout << "tf file changed" << endl;
    ModelTfWatcher* watcher = (ModelTfWatcher*)QObject::sender();
    if (!watcher) return;

    vector<string> folders = watcher->GetSubFolder();
    if (folders.empty()) return;
    vector<QTreeWidgetItem*> twItems = view->CreateJobItems(folders);
    

    // connect relation of job items , TfLiveInfo, and 

    QTreeWidget* tw = view->GetTwJobsTree();
    disconnect(tw, SIGNAL(itemClicked(QTreeWidgetItem*, int)),
                this, SLOT(handleTreeItemClicked(QTreeWidgetItem*, int)));
    connect(tw, SIGNAL(itemClicked(QTreeWidgetItem*, int)),
            this, SLOT(handleTreeItemClicked(QTreeWidgetItem*, int)));

    vector<TfLiveInfo*>* infos = watcher->GetLiveInfo();
    if (!infos) return;
    if (infos->empty()) return;

    // connect QTreeWidget with TfLiveInfo, ViewLineChartProps


    const TfTags tags;
    QStringList figs = {tags.tagEpochAcc.c_str(), tags.tagEpochLoss.c_str()};
    m_vChartSections.resize(figs.count());
    // iterate figs
    for (int i = 0; i < figs.count(); i++)
    {
        QWidget* sec = view->CreateChartSection(figs[i], NULL);
        m_vChartSections[i] = sec;
    }

    for (size_t i = 0;infos&& i < infos->size(); i++)
    {
        TfLiveInfo* info = infos->at(i);
        cout << *info << endl;
        ModelTfParser* parser = new ModelTfParser;
        parser->SetInputName(info->GetFileName());
        parser->start();
        parser->Wait();

        QVector<double>* epocshAcc = parser->GetEpochAcc();
        QVector<double>* epocshLoss = parser->GetEpochLoss();
        

        ChartInfo* ciAcc = new ChartInfo;
        ChartInfo* ciLoss = new ChartInfo;
        ciAcc->m_qvdXData.resize(epocshAcc->size());
        ciLoss->m_qvdXData.resize(epocshLoss->size());
        for (size_t j = 0; j < epocshAcc->size(); j++) {
            ciAcc->m_qvdXData[j] = j;
            ciLoss->m_qvdXData[j] = j;
        }
        ciAcc->m_qvdYData = *epocshAcc;
        ciLoss->m_qvdYData = *epocshLoss;
        ciAcc->m_sXLabel = "Step";
        ciLoss->m_sXLabel = "Step";
        ciAcc->m_sYLabel = tags.tagEpochAcc;
        ciLoss->m_sYLabel = tags.tagEpochLoss;

        ViewLineChartProps* vpAcc = (ViewLineChartProps*)m_vChartSections[0];
        if (vpAcc) {
            vpAcc->DrawLineChart(ciAcc);
        }
        ViewLineChartProps* vpLoss = (ViewLineChartProps*)m_vChartSections[1];
        if (vpLoss) {
            vpLoss->DrawLineChart(ciLoss);
        }
        
    }
    
}
