#ifndef LC_TENSORBOARD
#define LC_TENSORBOARD

#include "ViewTensorBoard.h"

class LcTensorBoard : public QObject
{
    Q_OBJECT
    public:
        LcTensorBoard(QObject *parent = NULL);
        ~LcTensorBoard();

    private:
        ViewTensorBoard *view=NULL;
};

#endif /* LC_TENSORBOARD */