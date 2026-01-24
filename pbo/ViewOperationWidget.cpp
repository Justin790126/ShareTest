#include "ViewOperationWidget.h"

#include <QComboBox>
#include <QLabel>
#include <QHBoxLayout>
#include <QPainter>
#include <QPen>

static void drawDiffRect_AminusB(QPainter& p,
                                 const QRect& A,
                                 const QRect& B,
                                 const QColor& fill)
{
    QRect I = A.intersected(B);
    if (I.isEmpty())
    {
        p.fillRect(A, fill);
        return;
    }

    if (I.top() > A.top())
        p.fillRect(QRect(A.left(), A.top(), A.width(), I.top() - A.top()), fill);

    if (I.bottom() < A.bottom())
        p.fillRect(QRect(A.left(), I.bottom() + 1, A.width(), A.bottom() - I.bottom()), fill);

    if (I.left() > A.left())
        p.fillRect(QRect(A.left(), I.top(), I.left() - A.left(), I.height()), fill);

    if (I.right() < A.right())
        p.fillRect(QRect(I.right() + 1, I.top(), A.right() - I.right(), I.height()), fill);
}

ViewOperationWidget::ViewOperationWidget(QWidget* parent)
    : QWidget(parent)
    , m_combo(0)
    , m_preview(0)
{
    QHBoxLayout* h = new QHBoxLayout(this);
    h->setContentsMargins(4, 2, 4, 2);
    h->setSpacing(10);

    m_combo = new QComboBox(this);
    m_combo->addItems(QStringList() << "AND" << "OR" << "XOR" << "DIFF");
    m_combo->setFixedWidth(90);
    m_combo->setFixedHeight(24);

    m_preview = new QLabel(this);
    m_preview->setAlignment(Qt::AlignCenter);
    m_preview->setMinimumWidth(200);
    m_preview->setFixedHeight(60);
    m_preview->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);

    h->addWidget(m_combo, 0, Qt::AlignVCenter);
    h->addWidget(m_preview, 1, Qt::AlignVCenter);

    connect(m_combo, SIGNAL(currentIndexChanged(QString)),
            this, SLOT(onComboChanged(QString)));

    refreshPixmap();
}

QString ViewOperationWidget::operation() const
{
    return m_combo ? m_combo->currentText() : QString("AND");
}

void ViewOperationWidget::setOperation(const QString& op)
{
    int idx = m_combo->findText(op);
    m_combo->setCurrentIndex(idx >= 0 ? idx : 0);
}

void ViewOperationWidget::onComboChanged(const QString&)
{
    refreshPixmap();
    emit operationChanged(operation());
}

void ViewOperationWidget::refreshPixmap()
{
    m_preview->setPixmap(makeOpPixmap(operation()));
}

QPixmap ViewOperationWidget::makeOpPixmap(const QString& op) const
{
    const int W = 200;
    const int H = 60;

    QPixmap pm(W, H);
    pm.fill(Qt::white);

    QPainter p(&pm);
    p.setRenderHint(QPainter::Antialiasing, true);

    QRect A(45, 12, 70, 36);
    QRect B(90, 12, 70, 36);

    QColor fill(120, 180, 255, 160);

    p.setPen(QPen(Qt::black, 2));
    p.setBrush(Qt::NoBrush);
    p.drawRect(A);
    p.drawRect(B);

    if (op == "AND")
    {
        QRect I = A.intersected(B);
        if (!I.isEmpty()) p.fillRect(I, fill);
    }
    else if (op == "OR")
    {
        p.fillRect(A, fill);
        p.fillRect(B, fill);
    }
    else if (op == "XOR")
    {
        p.fillRect(A, fill);
        p.fillRect(B, fill);
        QRect I = A.intersected(B);
        if (!I.isEmpty()) p.fillRect(I, Qt::white);
    }
    else if (op == "DIFF")
    {
        drawDiffRect_AminusB(p, A, B, fill);
    }

    p.setPen(QPen(Qt::black, 2));
    p.setBrush(Qt::NoBrush);
    p.drawRect(A);
    p.drawRect(B);

    return pm;
}
