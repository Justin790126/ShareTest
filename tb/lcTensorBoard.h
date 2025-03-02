#ifndef LC_TENSORBOARD
#define LC_TENSORBOARD

#include "ViewTensorBoard.h"
#include "ModelTfParser.h"
#include "utils.h"

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
        ModelTfWatcher* fsWatcher=NULL;
        Utils* utils=Utils::GetInstance();


    private slots:
        void handleTfFileChanged();
        void handleTreeItemClicked(QTreeWidgetItem* item, int idx);
};

#endif /* LC_TENSORBOARD */