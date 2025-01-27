#ifndef VIEW_TENSORBOARD
#define VIEW_TENSORBOARD

#include <iostream>
#include <string>
#include <vector>
#include <QtGui>

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
};

#endif /* VIEW_TENSORBOARD */