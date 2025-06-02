#include "ViewScalableItemText.h"

ViewScalableItemText::ViewScalableItemText(QCustomPlot* parentPlot)
    : QCPItemText(parentPlot), mScalingAxis(0), mBaseRange(1.0), mBaseFontPointSize(12)
{
    mBaseFontPointSize = font().pointSizeF();
}

void ViewScalableItemText::setScalingAxis(QCPAxis* axis, double baseRange)
{
    mScalingAxis = axis;
    mBaseRange = baseRange;
    mBaseFontPointSize = font().pointSizeF();
}

void ViewScalableItemText::draw(QCPPainter *painter)
{
    QFont scaledFont = font();
    if (mScalingAxis)
    {
        double scale = mBaseRange / mScalingAxis->range().size();
        scaledFont.setPointSizeF(mBaseFontPointSize * scale);
    }
    // print point size
    painter->setFont(scaledFont);

    QCPItemText::draw(painter);
}