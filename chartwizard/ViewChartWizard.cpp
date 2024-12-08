#include "ViewChartWizard.h"

ChartTypeDialog::ChartTypeDialog(QWidget *parent) : QDialog(parent), m_iChartTypeIdx(-1)
{
    Widgets();
    Layout();
    Connect();
}

void ChartTypeDialog::Widgets()
{
    btnOk = new QPushButton("OK");
    btnCancel = new QPushButton("Cancel");
}

void ChartTypeDialog::Connect()
{
    // connect cancel
    connect(btnCancel, SIGNAL(clicked()), this, SLOT(handleBtnCancelClicked()));
    // connect ok
    connect(btnOk, SIGNAL(clicked()), this, SLOT(handleBtnOkClicked()));

    // connect btngChartType
    connect(btngChartType, SIGNAL(buttonClicked(int)), this, SLOT(handleChartTypeChanged(int)));
}

void ChartTypeDialog::handleChartTypeChanged(int id)
{
    m_iChartTypeIdx = id;
}

void ChartTypeDialog::handleBtnCancelClicked()
{
    reject();
}

void ChartTypeDialog::handleBtnOkClicked()
{
    accept();
}

void ChartTypeDialog::Layout()
{
    QVBoxLayout *lytMain = new QVBoxLayout(this);
    {
        // add grid layout to lytMain
        QGridLayout *gridLayout = new QGridLayout();
        {
            btngChartType = new QButtonGroup();
            btngChartType->setExclusive(true);
            // add QToolButtons to gridLayout
            QToolButton *tlbtnLine = new QToolButton();
            tlbtnLine->setText("Line");
            tlbtnLine->setCheckable(true);
            tlbtnLine->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);

            QToolButton *tlbtnBar = new QToolButton();
            tlbtnBar->setText("Bar");
            tlbtnBar->setCheckable(true);
            tlbtnBar->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);

            QToolButton *tlbtnScatter = new QToolButton();
            tlbtnScatter->setText("Scatter");
            tlbtnScatter->setCheckable(true);
            tlbtnScatter->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);

            QToolButton *tlbtnPie = new QToolButton();
            tlbtnPie->setText("Pie");
            tlbtnPie->setCheckable(true);
            tlbtnPie->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);

            btngChartType->addButton(tlbtnLine, 0);
            btngChartType->addButton(tlbtnBar, 1);
            btngChartType->addButton(tlbtnScatter, 2);
            btngChartType->addButton(tlbtnPie, 3);

            gridLayout->addWidget(tlbtnLine, 0, 0);
            gridLayout->addWidget(tlbtnBar, 0, 1);
            gridLayout->addWidget(tlbtnScatter, 1, 0);
            gridLayout->addWidget(tlbtnPie, 1, 1);
        }
        lytMain->addLayout(gridLayout);

        QHBoxLayout *hboxButtons = new QHBoxLayout();
        {
            hboxButtons->addStretch();
            hboxButtons->addWidget(btnCancel);
            hboxButtons->addWidget(btnOk);
        }
        lytMain->addLayout(hboxButtons);
    }

    this->setLayout(lytMain);
}

PropsSection::PropsSection(const QString &title, const int animationDuration, QWidget *parent, int idx)
    : Section(title, animationDuration, parent), m_iSecIdx(idx)
{
    Widgets();
    Layout();
    Connect();
}

void PropsSection::Widgets()
{
}

void PropsSection::SetXData(const QVector<double> &data)
{
    m_xData = data;
    // get min/max of m_xData and set range for x-axis
    double minX = m_xData.front();
    double maxX = m_xData.back();
    for (double x : m_xData)
    {
        if (x < minX)
            minX = x;
        if (x > maxX)
            maxX = x;
    }
    GetQcp()->xAxis->setRange(minX, maxX);
}

void PropsSection::SetXLabel(const QString &lbl)
{
    GetQcp()->xAxis->setLabel(lbl);
    GetQcp()->replot();
}

void PropsSection::SetYData(const QVector<double> &data)
{
    m_yData = data;
    // get min/max of m_yData and set range for y-axis
    double minY = m_yData.front();
    double maxY = m_yData.back();
    for (double y : m_yData)
    {
        if (y < minY)
            minY = y;
        if (y > maxY)
            maxY = y;
    }
    GetQcp()->yAxis->setRange(minY, maxY);
}

void PropsSection::SetYLabel(const QString &lbl)
{
    GetQcp()->yAxis->setLabel(lbl);
    GetQcp()->replot();
}

void PropsSection::Connect()
{
    connect(cbbX, SIGNAL(currentIndexChanged(int)), this, SLOT(handleXAxisChanged(int)));
    connect(cbbY, SIGNAL(currentIndexChanged(int)), this, SLOT(handleYAxisChanged(int)));
    // connect leXaxis with SIGNAL(textChanged(const QString &)), this, SLOT(handleXaxisTextChanged(const QString &)));
    connect(leXaxis, SIGNAL(textChanged(const QString &)), this, SLOT(handleXaxisTextChanged(const QString &)));
    connect(leYaxis, SIGNAL(textChanged(const QString &)), this, SLOT(handleYaxisTextChanged(const QString &)));
}

void PropsSection::handleXaxisTextChanged(const QString &)
{
    emit xAxisTextChanged(leXaxis->text());
}

void PropsSection::handleYaxisTextChanged(const QString &)
{
    emit yAxisTextChanged(leYaxis->text());
}

void PropsSection::handleXAxisChanged(int idx)
{
    emit xAxisChanged(idx);
}

void PropsSection::handleYAxisChanged(int idx)
{
    emit yAxisChanged(idx);
}

void PropsSection::Plot()
{
    if (m_iSecIdx == 0)
    {
        GetGraph()->setData(m_xData, m_yData);
        GetQcp()->replot();
    }
}

void PropsSection::Layout()
{
    // QGroupBox *grpBox = new QGroupBox("Chart Properties", this);
    QVBoxLayout *anyLayout = new QVBoxLayout();
    anyLayout->setContentsMargins(0, 0, 0, 0);
    {
        QGroupBox *grpChartData = new QGroupBox("Trace", this);
        QFormLayout *fmlChartData = new QFormLayout(grpChartData);
        fmlChartData->setContentsMargins(0, 0, 0, 0);
        {
            cbbX = new QComboBox();
            cbbX->addItem("Col1");
            cbbX->addItem("Col2");
            cbbX->addItem("Col3");

            cbbY = new QComboBox();
            cbbY->addItem("Col1");
            cbbY->addItem("Col2");
            cbbY->addItem("Col3");

            leXaxis = new QLineEdit();
            leYaxis = new QLineEdit();

            fmlChartData->addRow(new QLabel("Title"), new QLineEdit());
            fmlChartData->addRow(new QLabel("X Data"), cbbX);
            fmlChartData->addRow(new QLabel("X Axis Label"), leXaxis);
            fmlChartData->addRow(new QLabel("Y Data"), cbbY);
            fmlChartData->addRow(new QLabel("Y Axis Label"), leYaxis);
        }
        QGroupBox *grpChartStyle = new QGroupBox("Chart Style", this);
        QVBoxLayout *vlytChartStyle = new QVBoxLayout(grpChartStyle);
        {
            // Add widgets to vlytChartStyle
            vlytChartStyle->addWidget(new QLabel("chart style"));
        }
        anyLayout->addWidget(grpChartData);
        anyLayout->addWidget(grpChartStyle);
    }

    this->setContentLayout(*anyLayout);
}

/*

 */

ViewChartWizard::ViewChartWizard(QWidget *parent) : QWidget(parent)
{
    Widgets();
    Layout();

    this->resize(800, 600);
    // connect(btnNewChart, SIGNAL(clicked()), this, SLOT(handleNewChartCreated()));
}

void ViewChartWizard::AddNewChart(PropsSection *newChart)
{
    vlytProps->insertWidget(1, newChart);
}

void ViewChartWizard::handleNewChartCreated()
{
    PropsSection *secChart = new PropsSection("Chart", 0);
    vlytProps->insertWidget(1, secChart);

    // PropsSection *secTrace = new PropsSection("Trace", 0);
    // vlytProps->insertWidget(1, secTrace);
}

void ViewChartWizard::CreateTableWidget()
{
    tbwCSVcontent = new QTableWidget(this);
    widTable = new QWidget(this);
    {
        widTable->setLayout(new QVBoxLayout());
        // Add widgets to widTable
        widTable->layout()->addWidget(new QLabel("Table"));
        widTable->layout()->addWidget(tbwCSVcontent);
    }
}

void ViewChartWizard::CreatePropWidget()
{
    btnNewChart = new QPushButton("New Chart");
    btnNewChart->setFixedWidth(100);

    widProps = new QWidget(this);

    vlytProps = new QVBoxLayout();
    {
        vlytProps->addWidget(btnNewChart);
        vlytProps->addStretch(1);
    }
    widProps->setLayout(vlytProps);
    sclProps = new QScrollArea;
    sclProps->setWidget(widProps);
    sclProps->setWidgetResizable(true);
}

void ViewChartWizard::Widgets()
{
    CreatePropWidget();
    CreateTableWidget();

    // widChart = new QWidget(this);
    // {
    //     widChart->setLayout(new QVBoxLayout());
    //     // Add widgets to widChart
    //     widChart->layout()->addWidget(new QLabel("Chart"));
    // }
    qcp = new QCustomPlot();
    qcp->legend->setVisible(true);
   
    // create two new graphs and set their look:
    // QCPGraph *graph1 = qcp->addGraph();
    // clear QCPGraph

    // QCPGraph *graph = qcp->addGraph();
    // QVector<double> x(101), y(101); // initialize with entries 0...100
    // for (int i=0; i<101; ++i)
    // {
    //   x[i] = i/100.0; // x goes from 0 to 1
    //   y[i] = qSin(x[i]*2*M_PI); // let's plot a sine curve
    // }
    // graph->setData(x, yy);
    // qcp->xAxis->setLabel("x");
    // qcp->yAxis->setLabel("sin(x)");
    // qcp->xAxis->setRange(0, 1);
    // qcp->yAxis->setRange(-1.1, 1.1);
    // qcp->replot();
}

void ViewChartWizard::Layout()
{
    QVBoxLayout *vlytMain = new QVBoxLayout(this);
    vlytMain->setContentsMargins(0, 0, 0, 0);
    {
        QHBoxLayout *hlytMain = new QHBoxLayout();
        hlytMain->setContentsMargins(0, 0, 0, 0);
        {
            QSplitter *splMain = new QSplitter(Qt::Horizontal, this);
            {
                splMain->addWidget(sclProps);
                {
                    splMain->addWidget(qcp);
                }
                // splMain->addWidget(widVisual);
            }
            hlytMain->addWidget(splMain);
            splMain->setSizes({200, 600});
        }
        vlytMain->addLayout(hlytMain);
    }

    this->setLayout(vlytMain);
}