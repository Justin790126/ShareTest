#ifndef LC_TENSORBOARD
#define LC_TENSORBOARD

#include "ViewTensorBoard.h"
#include "ModelTfParser.h"

struct TbArgs
{
    string m_sLogDir="";
};


static TbArgs tbArgs;

class LcTensorBoard : public QObject
{
    Q_OBJECT
    public:
        LcTensorBoard(TbArgs args, QObject *parent = NULL);
        ~LcTensorBoard();

    private:
        ViewTensorBoard *view=NULL;
        ModelTfParser* model=NULL;
};

#endif /* LC_TENSORBOARD */