#include "lcTensorBoard.h"

LcTensorBoard::LcTensorBoard(QObject *parent)
    : QObject(parent)
{
    
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

