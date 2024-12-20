#ifndef LC_XML_H
#define LC_XML_H

#include <QtGui>
// write class inherit QWidget
#include "ViewXmlTab.h"
#include "ModelXmlParser.h"

class LcXml : public QWidget
{
    Q_OBJECT
    public:
        LcXml(QWidget *parent=NULL);
        ~LcXml() = default;

    private:
        ViewXmlTab * m_vtXmlTab = NULL;
        ModelXmlParser* m_mXmlParser = NULL;
    
    private slots:
        void handleAllPageReaded();
};


#endif /* LC_XML_H */