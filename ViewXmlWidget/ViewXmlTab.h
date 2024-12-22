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

        QTreeWidget* GetTwXmlViewer() const { return twXmlViewer; }


    private:

        void Widgets();
        void Layout();
        void UI();

        QTreeWidget * twXmlViewer = NULL;
};


#endif /* VIEW_XML_TAB_H */