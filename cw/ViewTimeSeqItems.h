#ifndef VIEW_TIME_SEQ_ITEMS_H
#define VIEW_TIME_SEQ_ITEMS_H

#include "qcustomplot.h"
#include <iostream>
#include <iomanip>

using namespace std;

class ViewTimeSeqBar : public QCPItemRect
{
    Q_OBJECT
public:
    ViewTimeSeqBar(QCustomPlot *parentPlot, int iActType, int iActId)
        : QCPItemRect(parentPlot), m_iActType(iActType), m_iActId(iActId) {
        setPen(QPen(Qt::black));
        setBrush(QBrush(Qt::blue));
    }

    ~ViewTimeSeqBar() = default;

    int GetActType() const { return m_iActType; }
    int GetActId() const { return m_iActId; }
    void SetActType(int iActType) { m_iActType = iActType; }
    void SetActId(int iActId) { m_iActId = iActId; }
    void setBarPosition(double x1, double y1, double x2, double y2) {
        topLeft->setCoords(x1, y1);
        bottomRight->setCoords(x2, y2);
    }
    
    friend ofstream& operator<<(ofstream& os, const ViewTimeSeqBar& bar) {
        printf("ViewTimeSeqBar: ActType=%d, ActId=%d\n", bar.m_iActType, bar.m_iActId);
        return os;
    }
private:
    int m_iActType;       // Action type (e.g., send, receive)
    int m_iActId;         // Action ID (e.g., enable profile, load model)
};

#endif /* VIEW_TIME_SEQ_ITEMS */