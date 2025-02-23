#include "lcTensorBoard.h"

LcTensorBoard::LcTensorBoard(TbArgs args, QObject *parent)
    : QObject(parent)
{
    tbArgs = args;

    // feed logdir to ModelTfWatcher
    fsWatcher = new ModelTfWatcher;
    if (!tbArgs.m_sLogDir.empty()) {
        fsWatcher->SetLogDir(tbArgs.m_sLogDir);
        fsWatcher->start();
    }
    

    if (!view) {
        view = new ViewTensorBoard();
        view->show();
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

