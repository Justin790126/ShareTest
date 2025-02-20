#include "lcTensorBoard.h"

LcTensorBoard::LcTensorBoard(TbArgs args, QObject *parent)
    : QObject(parent)
{
    tbArgs = args;

    // feed logdir to ModelTfParser
    model = new ModelTfParser;
    model->SetLogDir(tbArgs.m_sLogDir);
    model->start();
    model->Wait();

    if (!view) {
        view = new ViewTensorBoard();
        view->show();
    }
}

LcTensorBoard::~LcTensorBoard()
{
    if (view) delete view;
    view = NULL;
}

