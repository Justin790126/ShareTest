#include "BoolTableWidget.h"

#include <QTableWidget>
#include <QHeaderView>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QComboBox>
#include <QFileDialog>
#include <QPainter>
#include <QPen>

// ---------- helper: file cell ----------
static QWidget* makeFileCell(QWidget* parent,
                            const QString& buttonText,
                            QPushButton** outBtn,
                            QLabel** outLabel)
{
    QWidget* cell = new QWidget(parent);
    QVBoxLayout* v = new QVBoxLayout(cell);
    v->setContentsMargins(4, 4, 4, 4);
    v->setSpacing(4);

    QPushButton* btn = new QPushButton(buttonText, cell);

    QLabel* label = new QLabel("-", cell);
    label->setWordWrap(true);
    label->setMinimumHeight(36);
    label->setTextInteractionFlags(Qt::TextSelectableByMouse);

    v->addWidget(btn);
    v->addWidget(label);

    if (outBtn) *outBtn = btn;
    if (outLabel) *outLabel = label;
    return cell;
}

// ---------- ctor ----------
BoolTableWidget::BoolTableWidget(QWidget* parent)
    : QWidget(parent)
    , m_table(0)
    , m_leftPathLabel(0)
    , m_rightPathLabel(0)
    , m_combo(0)
    , m_opImageLabel(0)
{
    // 3 columns now
    m_table = new QTableWidget(1, 3, this);
    m_table->setHorizontalHeaderLabels(
        QStringList() << "Input A" << "Operation" << "Input B"
    );

    m_table->verticalHeader()->hide();
    m_table->setSelectionMode(QAbstractItemView::NoSelection);
    m_table->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_table->setFocusPolicy(Qt::NoFocus);

    m_table->horizontalHeader()->setResizeMode(0, QHeaderView::Stretch);
    m_table->horizontalHeader()->setResizeMode(1, QHeaderView::ResizeToContents);
    m_table->horizontalHeader()->setResizeMode(2, QHeaderView::Stretch);

    m_table->setRowHeight(0, 160);
    m_table->setColumnWidth(1, 300);

    // ---- col0: input A ----
    QPushButton* leftBtn = 0;
    QWidget* leftCell = makeFileCell(m_table, "Open...", &leftBtn, &m_leftPathLabel);
    m_table->setCellWidget(0, 0, leftCell);
    connect(leftBtn, SIGNAL(clicked()), this, SLOT(openFileLeft()));

    // ---- col1: merged operation cell ----
    QWidget* opCell = new QWidget(m_table);
    QVBoxLayout* opLayout = new QVBoxLayout(opCell);
    opLayout->setContentsMargins(4, 4, 4, 4);
    opLayout->setSpacing(6);

    m_combo = new QComboBox(opCell);
    m_combo->addItems(QStringList() << "AND" << "OR" << "XOR" << "DIFF");
    connect(m_combo, SIGNAL(currentIndexChanged(QString)),
            this, SLOT(operationChanged(QString)));

    m_opImageLabel = new QLabel(opCell);
    m_opImageLabel->setAlignment(Qt::AlignCenter);
    m_opImageLabel->setMinimumSize(260, 120);

    opLayout->addWidget(m_combo);
    opLayout->addWidget(m_opImageLabel, 1);

    m_table->setCellWidget(0, 1, opCell);

    // ---- col2: input B ----
    QPushButton* rightBtn = 0;
    QWidget* rightCell = makeFileCell(m_table, "Open...", &rightBtn, &m_rightPathLabel);
    m_table->setCellWidget(0, 2, rightCell);
    connect(rightBtn, SIGNAL(clicked()), this, SLOT(openFileRight()));

    // ---- bottom buttons ----
    QPushButton* okBtn = new QPushButton("OK", this);
    QPushButton* cancelBtn = new QPushButton("Cancel", this);
    connect(okBtn, SIGNAL(clicked()), this, SLOT(onOk()));
    connect(cancelBtn, SIGNAL(clicked()), this, SLOT(onCancel()));

    QHBoxLayout* bottom = new QHBoxLayout;
    bottom->addStretch(1);
    bottom->addWidget(okBtn);
    bottom->addWidget(cancelBtn);

    QVBoxLayout* main = new QVBoxLayout(this);
    main->addWidget(m_table);
    main->addLayout(bottom);

    refreshOpPixmap();
}

// ---------- slots ----------
void BoolTableWidget::openFileLeft()
{
    QString path = QFileDialog::getOpenFileName(this, "Open input A");
    if (path.isEmpty()) return;

    m_leftPath = path;
    m_leftPathLabel->setText(path);
}

void BoolTableWidget::openFileRight()
{
    QString path = QFileDialog::getOpenFileName(this, "Open input B");
    if (path.isEmpty()) return;

    m_rightPath = path;
    m_rightPathLabel->setText(path);
}

void BoolTableWidget::operationChanged(const QString&)
{
    refreshOpPixmap();
}

void BoolTableWidget::onOk()
{
    emit accepted(m_leftPath, m_combo->currentText(), m_rightPath);
    close();
}

void BoolTableWidget::onCancel()
{
    emit rejected();
    close();
}

// ---------- pixmap ----------
void BoolTableWidget::refreshOpPixmap()
{
    m_opImageLabel->setPixmap(makeOpPixmap(m_combo->currentText()));
}

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
        p.fillRect(QRect(A.left(), A.top(),
                          A.width(), I.top() - A.top()), fill);

    if (I.bottom() < A.bottom())
        p.fillRect(QRect(A.left(), I.bottom() + 1,
                          A.width(), A.bottom() - I.bottom()), fill);

    if (I.left() > A.left())
        p.fillRect(QRect(A.left(), I.top(),
                          I.left() - A.left(), I.height()), fill);

    if (I.right() < A.right())
        p.fillRect(QRect(I.right() + 1, I.top(),
                          A.right() - I.right(), I.height()), fill);
}

QPixmap BoolTableWidget::makeOpPixmap(const QString& op) const
{
    QPixmap pm(260, 120);
    pm.fill(Qt::white);

    QPainter p(&pm);
    p.setRenderHint(QPainter::Antialiasing);

    QRect A(60, 30, 80, 60);
    QRect B(110, 30, 80, 60);

    QColor fill(120, 180, 255, 160);

    // outlines
    p.setPen(QPen(Qt::black, 2));
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
    p.drawRect(A);
    p.drawRect(B);

    p.drawText(pm.rect(), Qt::AlignBottom | Qt::AlignHCenter, op);

    return pm;
}
