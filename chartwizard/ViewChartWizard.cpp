#include "ViewChartWizard.h"

ChartTypeDialog::ChartTypeDialog(QWidget *parent) : QDialog(parent)
{
    
}

PropsSection::PropsSection(const QString &title, const int animationDuration, QWidget *parent, int idx)
    : Section(title, animationDuration, parent), m_iSecIdx(idx)
{
    Widgets();
    Layout();
}

void PropsSection::Widgets()
{
}

void PropsSection::Layout()
{
    // QGroupBox *grpBox = new QGroupBox("Chart Properties", this);
    QVBoxLayout *anyLayout = new QVBoxLayout();
    anyLayout->setContentsMargins(0, 0, 0, 0);
    {
        QGroupBox* grpChartData = new QGroupBox("Trace", this);
        QFormLayout *fmlChartData = new QFormLayout(grpChartData);
        fmlChartData->setContentsMargins(0, 0, 0, 0);
        {
            fmlChartData->addRow(new QLabel("Trace Name"), new QLineEdit());
            // fmlChartData->addRow(new QLabel("Trace Name"), new("Line Chart"));
        }
        QGroupBox* grpChartStyle = new QGroupBox("Chart Style", this);
        QVBoxLayout* vlytChartStyle = new QVBoxLayout(grpChartStyle);
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

void ViewChartWizard::AddNewChart(PropsSection* newChart)
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
    // QCPGraph *graph = qcp->addGraph();
    // QVector<double> x(101), y(101); // initialize with entries 0...100
    // for (int i=0; i<101; ++i)
    // {
    //   x[i] = i/100.0; // x goes from 0 to 1
    //   y[i] = qSin(x[i]*2*M_PI); // let's plot a sine curve
    // }
    // graph->setData(x, y);
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