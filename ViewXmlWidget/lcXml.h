#ifndef LC_XML_H
#define LC_XML_H

#include <QtGui>
// write class inherit QWidget
#include "ViewXmlTab.h"

class LcXml : public QWidget
{
    Q_OBJECT
    public:
        LcXml(QWidget *parent=NULL);
        ~LcXml() = default;

    private:
        ViewXmlTab * m_vtXmlTab = NULL;
};


#endif /* LC_XML_H */