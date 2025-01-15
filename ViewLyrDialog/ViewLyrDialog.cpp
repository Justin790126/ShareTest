#include "ViewLyrDialog.h"

#include<QHBoxLayout>
#include<QPainter>
#include<QPaintEvent>
#include<QPoint>
#include<QLine>
QPenWidth::QPenWidth(QWidget  *parent)
    : QLabel(parent)
    , m_drawType(0)
    , m_size(1)
{
}

QPenWidth::~QPenWidth()
{
}

void QPenWidth::SetDrawParameters(int drawType, int isize)
{
    m_drawType = drawType;
    m_size = isize;
}

void QPenWidth::paintEvent(QPaintEvent *)
{
    QSize size = this->size();
    QPainter painter(this);
    QPen pen;
    pen.setColor(Qt::black);
    pen.setWidth(m_size);
    painter.setPen(pen);
    if (m_drawType == 0)
        painter.drawPoint(QPoint(size.width() / 2, size.height() / 2));
    else
    {

        QPoint p1(0, size.height() / 2);
        QPoint p2(size.width(), size.height() / 2);
        QLine line(p1, p2);
        painter.drawLine(line);
    }

}

QPenWidget::QPenWidget(QWidget *parent /*= NULL*/)
    :QLineEdit(parent)
    , m_index(0)
    , m_type(0)
{
    m_pLabel = new QLabel();
    m_pCssLabel = new QPenWidth();

    m_pLabel->setFixedSize(12, 12);

    QHBoxLayout* layout = new QHBoxLayout();
    layout->addWidget(m_pLabel);
    layout->addWidget(m_pCssLabel);
    layout->setContentsMargins(5, 0, 0, 2);
    setLayout(layout);
    setReadOnly(true);

    setStyleSheet("QLineEdit{border: none;}QLineEdit:hover{background-color:rgb(0,125,255);}");

}

QPenWidget::~QPenWidget()
{

}

void QPenWidget::updatePen(const int& index, const int& type)
{
    m_index = index;
    m_type = type;

    QString strText = QString("%1 )").arg(QString::number(m_index));
    m_pLabel->setText(strText);
    m_pCssLabel->SetDrawParameters(type, index);

}

void QPenWidget::mousePressEvent(QMouseEvent *event)
{
    emit click(m_index);
}

QLineWidthCombobox::QLineWidthCombobox(QWidget *parent /*= NULL*/)
    :QComboBox(parent)
    , m_drawType(1)
{
    m_pPenEdit = new QPenWidget();
    m_pListWidget = new QListWidget();
    m_pPenEdit->setStyleSheet("");
    setContextMenuPolicy(Qt::NoContextMenu);//禁用菜单
    m_pListWidget->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);//禁用垂直滚动条
    m_pListWidget->setStyleSheet("QListWidget::Item:hover{background-color:rgb(0,125,255);}");
    setLineEdit(m_pPenEdit);
    setModel(m_pListWidget->model());
    setView(m_pListWidget);
}

QLineWidthCombobox::~QLineWidthCombobox()
{

}

void QLineWidthCombobox::SetType(int drawType)
{
    m_drawType = drawType;
}

int QLineWidthCombobox::GetCurrentIndex()
{
    return m_index;
}

void QLineWidthCombobox::SetCurrentIndex(int index)
{
    m_index = index;
    m_pPenEdit->updatePen(index, m_drawType);
}

void QLineWidthCombobox::SetList(QList<int>&list)
{
    m_list = list;
    m_pListWidget->clear();
    int icount = m_list.count();
    for (int i = 0; i < icount; i++)
        appendItem(m_list[i]);
}

QList<int> QLineWidthCombobox::GetList()
{
    return m_list;
}

void QLineWidthCombobox::onClickPenWidget(const int& index)
{
    m_index = index;
    m_pPenEdit->updatePen(index, m_drawType);
    hidePopup();
    emit SelectedItemChanged(m_index);
}

void QLineWidthCombobox::appendItem(const int& index)
{
    QPenWidget* pWid = new QPenWidget(this);
    pWid->updatePen(index, m_drawType);
    connect(pWid, SIGNAL(click(const int&)), this, SLOT(onClickPenWidget(const int&)));
    QListWidgetItem* pItem = new QListWidgetItem(m_pListWidget);

    m_pListWidget->addItem(pItem);
    m_pListWidget->setItemWidget(pItem, pWid);
}

ViewLyrDialog::ViewLyrDialog(QWidget *parent) : QDialog(parent)
{
    Widgets();
    Layout();
}

void ViewLyrDialog::Widgets()
{
    leLyrName = new QLineEdit();
    leLyrNum = new QLineEdit();
    leLyrNum->setReadOnly(true);
    leLyrDType = new QLineEdit();
    leLyrDType->setReadOnly(true);

    btnFillMoreColors = new QPushButton("More Colors...");
    leFillColorHex = new QLineEdit();
    // set line edit background blue, white text
    leFillColorHex->setStyleSheet("background-color: blue; color: white;");
    sldFillColorAlpha = new QSlider(Qt::Horizontal);
    lblFillAlphaValue = new QLabel("40%");

    btnOutlineMoreColors = new QPushButton("More Colors...");
    leOutlineColorHex = new QLineEdit();
    // set line edit background blue, white text
    leOutlineColorHex->setStyleSheet("background-color: blue; color: white;");
    sldOutlineColorAlpha = new QSlider(Qt::Horizontal);
    lblOutlineAlphaValue = new QLabel("40%");

    cbbOutlineWidth = new QLineWidthCombobox();
    cbbOutlineWidth->appendItem(1);
    cbbOutlineWidth->appendItem(2);
    cbbOutlineWidth->appendItem(3);
    cbbOutlineWidth->appendItem(4);
    cbbOutlineWidth->appendItem(5);
    spbOutlineWidth = new QSpinBox();

    btnOK = new QPushButton("OK");
    btnCancel = new QPushButton("Cancel");
    btnApply = new QPushButton("Apply");
}

void ViewLyrDialog::Layout()
{
    QVBoxLayout *vlytMain = new QVBoxLayout(this);
    vlytMain->setContentsMargins(15, 10, 15, 10);
    {
        QHBoxLayout *hlytLyrName = new QHBoxLayout();
        {
            hlytLyrName->addWidget(new QLabel("Name"), 1);
            hlytLyrName->addWidget(CreateSeparator(), 9);
        }
        QHBoxLayout *hlytLyrNumDtype = new QHBoxLayout();
        hlytLyrNumDtype->setContentsMargins(10, 5, 10, 5);
        {
            hlytLyrNumDtype->addWidget(new QLabel("Layer:"));
            hlytLyrNumDtype->addWidget(leLyrNum, 3);
            hlytLyrNumDtype->addWidget(new QLabel(" : "));
            hlytLyrNumDtype->addWidget(leLyrDType, 3);
        }
        QHBoxLayout *hlytLyrNameEdit = new QHBoxLayout();
        hlytLyrNameEdit->setContentsMargins(10, 5, 10, 5);
        {
            hlytLyrNameEdit->addWidget(new QLabel("Name:"), 1);
            hlytLyrNameEdit->addWidget(leLyrName, 9);
        }

        QHBoxLayout *hlytFillColorPattern = new QHBoxLayout();
        {
            hlytFillColorPattern->addWidget(new QLabel("Fill Color and Pattern"), 1);
            hlytFillColorPattern->addWidget(CreateSeparator(), 9);
        }
        QHBoxLayout *hlytFillColor = new QHBoxLayout();
        {
            QVBoxLayout *vlytFillColor = new QVBoxLayout();
            {
                QGridLayout *gdlFillColorLayout = new QGridLayout();
                {
                    // add 3 row x 9 columns color
                    for (int i = 0; i < 3; ++i)
                    {
                        for (int j = 0; j < 9; ++j)
                        {
                            QPushButton *btnColor = new QPushButton();
                            btnColor->setFixedSize(20, 20);
                            // btnColor->setFlat(true);
                            // add random color
                            btnColor->setStyleSheet(QString("background-color: %1").arg(QColor::fromHslF(qrand() / static_cast<double>(RAND_MAX), 1.0, 0.5).name()));
                            // add button to layout
                            gdlFillColorLayout->addWidget(btnColor, i, j);
                        }
                    }
                }

                QHBoxLayout *hlytFillColorAlpha = new QHBoxLayout();
                {
                    hlytFillColorAlpha->addWidget(new QLabel("Icon 0%"), 1);
                    hlytFillColorAlpha->addWidget(sldFillColorAlpha, 7);
                    hlytFillColorAlpha->addWidget(lblFillAlphaValue, 2);
                    hlytFillColorAlpha->addWidget(new QLabel("Icon 100%"), 1);
                }
                vlytFillColor->addLayout(gdlFillColorLayout);
                vlytFillColor->addLayout(hlytFillColorAlpha);
                vlytFillColor->addStretch(2);
            }
            QVBoxLayout *vlytMoreColors = new QVBoxLayout();
            {
                vlytMoreColors->addWidget(btnFillMoreColors);
                vlytMoreColors->addWidget(leFillColorHex);
                vlytMoreColors->addStretch(2);
            }
            hlytFillColor->addLayout(vlytFillColor, 6);
            hlytFillColor->addLayout(vlytMoreColors, 4);
        }
        QHBoxLayout *hlytOutlineColorStyleWidth = new QHBoxLayout();
        {
            hlytOutlineColorStyleWidth->addWidget(new QLabel("Outline Color, Style, and Width"), 1);
            hlytOutlineColorStyleWidth->addWidget(CreateSeparator(), 9);
        }
        QHBoxLayout *hlytOutlineColor = new QHBoxLayout();
        {
            QVBoxLayout *vlytOutlineColorStyleWidth = new QVBoxLayout();
            {
                QGridLayout *gdlOutlineColorStyleWidthLayout = new QGridLayout();
                {
                    // add 3 row x 9 columns color
                    for (int i = 0; i < 3; ++i)
                    {
                        for (int j = 0; j < 9; ++j)
                        {
                            QPushButton *btnColor = new QPushButton();
                            btnColor->setFixedSize(20, 20);
                            // btnColor->setFlat(true);
                            // add random color
                            btnColor->setStyleSheet(QString("background-color: %1").arg(QColor::fromHslF(qrand() / static_cast<double>(RAND_MAX), 1.0, 0.5).name()));
                            // add button to layout
                            gdlOutlineColorStyleWidthLayout->addWidget(btnColor, i, j);
                        }
                    }
                }

                vlytOutlineColorStyleWidth->addLayout(gdlOutlineColorStyleWidthLayout);
                vlytOutlineColorStyleWidth->addStretch(1);
            }
            QVBoxLayout *vlytMoreColors = new QVBoxLayout();
            {
                vlytMoreColors->addWidget(btnOutlineMoreColors);
                vlytMoreColors->addWidget(leOutlineColorHex);
                vlytMoreColors->addStretch(1);
            }
            hlytOutlineColor->addLayout(vlytOutlineColorStyleWidth, 6);
            hlytOutlineColor->addLayout(vlytMoreColors, 4);
        }
        QHBoxLayout *hlytOutlineWidth = new QHBoxLayout();
        {
            hlytOutlineWidth->addWidget(cbbOutlineWidth);
            hlytOutlineWidth->addWidget(spbOutlineWidth);
            hlytOutlineWidth->addWidget(new QLabel("pixels"));
        }
        QHBoxLayout *hlytOutlineColorStyleWidthAlpha = new QHBoxLayout();
        {
            hlytOutlineColorStyleWidthAlpha->addWidget(new QLabel("Icon 0%"), 1);
            hlytOutlineColorStyleWidthAlpha->addWidget(sldOutlineColorAlpha, 2);
            hlytOutlineColorStyleWidthAlpha->addWidget(lblFillAlphaValue, 1);
            hlytOutlineColorStyleWidthAlpha->addWidget(new QLabel("Icon 100%"), 1);
            hlytOutlineColorStyleWidthAlpha->addStretch(4);
        }

        QHBoxLayout *hlytBtns = new QHBoxLayout();
        {
            hlytBtns->addWidget(btnOK);
            hlytBtns->addWidget(btnCancel);
            hlytBtns->addWidget(btnApply);
        }

        vlytMain->addLayout(hlytLyrName);
        vlytMain->addLayout(hlytLyrNumDtype);
        vlytMain->addLayout(hlytLyrNameEdit);
        vlytMain->addLayout(hlytFillColorPattern);
        vlytMain->addLayout(hlytFillColor);
        vlytMain->addLayout(hlytOutlineColorStyleWidth);
        vlytMain->addLayout(hlytOutlineColor);
        vlytMain->addLayout(hlytOutlineWidth);
        vlytMain->addLayout(hlytOutlineColorStyleWidthAlpha);
        vlytMain->addWidget(CreateSeparator());
        vlytMain->addLayout(hlytBtns);
    }
    this->setLayout(vlytMain);
    this->resize(350, 500);
}

QFrame *ViewLyrDialog::CreateSeparator()
{
    QFrame *separator = new QFrame(this);
    separator->setFrameShape(QFrame::HLine);
    separator->setFrameShadow(QFrame::Sunken);

    return separator;
}