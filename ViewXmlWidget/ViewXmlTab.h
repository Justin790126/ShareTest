#ifndef VIEW_XML_TAB_H
#define VIEW_XML_TAB_H

#include <QtGui>

/* Aim to visualize Xml */

class ViewXmlTab : public QWidget
{
    Q_OBJECT
    public:
        ViewXmlTab(QWidget *parent=NULL);
        ~ViewXmlTab() = default;

        QTreeWidget* GettwXmlViewerLeft() const { return twXmlViewerLeft; }
        QTreeWidget* GettwXmlViewerRight() const { return twXmlViewerRight; }


    private:

        void Widgets();
        void Layout();
        void UI();
        void Connect();

        QTreeWidget * twXmlViewerLeft = NULL;
        QTreeWidget * twXmlViewerRight = NULL;
        QListWidget* lwDiffSummary = NULL;
    
    private slots:
        void handleClipItemToClipBoard();
};


#endif /* VIEW_XML_TAB_H */