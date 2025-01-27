#include "ViewTensorBoard.h"

ViewTensorBoard::ViewTensorBoard(QWidget *parent)
    : QWidget(parent)
{
    UI();
}

void ViewTensorBoard::widgets()
{
}

void ViewTensorBoard::layouts()
{
}

void ViewTensorBoard::UI()
{
    widgets();
    layouts();

    // Set up the window
    setWindowTitle(tr("TensorBoard"));
    // setMinimumSize(800, 600);
    
}