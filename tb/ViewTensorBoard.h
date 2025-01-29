#ifndef VIEW_TENSORBOARD
#define VIEW_TENSORBOARD

#include <iostream>
#include <string>
#include <vector>
#include <QtGui>
#include "ViewTimeSeries.h"

using namespace std;

class ViewTensorBoard : public QWidget
{
    Q_OBJECT
public:
    ViewTensorBoard(QWidget *parent = NULL);
    ~ViewTensorBoard() = default;

private:
    void widgets();
    void layouts();
    void UI();

    QTabWidget* tbwMain;

    ViewTimeSeries* viewTimeSeries;
};

#endif /* VIEW_TENSORBOARD */