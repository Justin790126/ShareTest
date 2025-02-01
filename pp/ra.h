#ifndef RA_H
#define RA_H

#include <QtGui>
#include <iostream>
using namespace std;

class Ra : public QWidget
{
    Q_OBJECT
    public:
        Ra(QWidget *parent=NULL);
        ~Ra() = default;

        void SetImage(QImage* pImg) { m_pImg = pImg; }
    protected:
        void paintEvent(QPaintEvent* event) override;
        void keyPressEvent(QKeyEvent* event) override;
    
    private:
        QImage* m_pImg = NULL;

};

#endif /* RA_H */