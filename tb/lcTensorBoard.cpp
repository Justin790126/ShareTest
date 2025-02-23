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


void LcTensorBoard::handleTfFileChanged()
{
    cout << "tf file changed" << endl;
    ModelTfWatcher* watcher = (ModelTfWatcher*)QObject::sender();
    if (!watcher) return;

    vector<TfLiveInfo*>* infos = watcher->GetLiveInfo();
    for (size_t i = 0;infos&& i < infos->size(); i++)
    {
        TfLiveInfo* info = infos->at(i);
        cout << *info << endl;
        ModelTfParser* parser = new ModelTfParser;
        parser->SetInputName(info->GetFileName());
        parser->start();
    }
    
}
