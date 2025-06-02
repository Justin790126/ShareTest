#ifndef VIEWSCALABLEITEMTEXT_H
#define VIEWSCALABLEITEMTEXT_H

#include "qcustomplot.h"

class ViewScalableItemText : public QCPItemText
{
public:
    explicit ViewScalableItemText(QCustomPlot* parentPlot);

    // Set the base (unzoomed) font size and axis for scaling
    void setScalingAxis(QCPAxis* axis, double baseRange);

protected:
    virtual void draw(QCPPainter *painter);

private:
    QCPAxis* mScalingAxis;
    double mBaseRange;
    double mBaseFontPointSize;
};

#endif // VIEWSCALABLEITEMTEXT_H