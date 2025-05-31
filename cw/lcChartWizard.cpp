#include "lcChartWizard.h"

lcChartWizard::lcChartWizard(QWidget* parent)
    : QWidget(parent)
{
    UI();

    resize(640, 480);

    vector<double> x_nms(SLICE_DATA.size());
    double nmPerStep = 4;
    for (int i = 0; i < SLICE_DATA.size(); ++i)
    {
        x_nms[i] = i*nmPerStep; // Assuming x-axis is just the index
    }
    vector<double> y_noise(SLICE_DATA.size());
    for (int i = 0; i < SLICE_DATA.size(); ++i)
    {
        // add random noise
        y_noise[i] = SLICE_DATA[i] + ((rand() % 100) / 10000.0); // Adding small random noise
    }

    // draw line chart
    // m_qcp->addGraph();
    // m_qcp->graph(0)->setData(QVector<double>::fromStdVector(x_nms), QVector<double>::fromStdVector(y_noise));
    // m_qcp->xAxis->setLabel("X Axis (nm)");
    // m_qcp->yAxis->setLabel("Y Axis (arbitrary units)");
    // m_qcp->xAxis->setRange(0, x_nms.back());
    // m_qcp->yAxis->setRange(*std::min_element(y_noise.begin(), y_noise.end()), *std::max_element(y_noise.begin(), y_noise.end()));
    // m_qcp->replot();
    // // let plot selectable
    // m_qcp->setInteractions(QCP::iSelectPlottables | QCP::iSelectAxes | QCP::iRangeDrag | QCP::iRangeZoom);

    // // change line color
    // m_qcp->graph(0)->setPen(QPen(Qt::yellow, 2)); // Set line color to blue and width to 2
    // // change line brush
    // m_qcp->graph(0)->setBrush(QBrush(QColor(255, 255, 0, 50))); // Set fill color with some transparency
    // // scatter points with circle
    // m_qcp->graph(0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCircle, 5)); // Set scatter style to circle with radius 5
    // // set name
    // m_qcp->graph(0)->setName("Line Chart ExampleQQ");
    // // set title
    // // m_qcp->plotLayout()->insertRow(0);
    // // m_qcp->plotLayout()->addElement(0, 0, new QCPTextElement(m_qcp, "Line Chart Example", QFont("sans", 12, QFont::Bold)));
    // // set legend
    // m_qcp->legend->setVisible(true);
    // m_qcp->legend->setFont(QFont("sans", 10));
    // m_qcp->legend->setBrush(QBrush(QColor(255, 255, 255, 200))); // Set legend background color with some transparency
    // update y axis data
    // m_qcp->graph(0)->setData(QVector<double>::fromStdVector(x_nms), QVector<double>::fromStdVector(SLICE_DATA));
    // m_qcp->replot();
}

lcChartWizard::~lcChartWizard()
{
    // Destructor implementation
}

void lcChartWizard::Widgets()
{
    // Initialize widgets here
    

}

QWidget* lcChartWizard::CreateLineChartProps(ChartInfo& chartInfo)
{
    ViewLineChartProps* section = new ViewLineChartProps("Line Chart Properties");
    {
        QVBoxLayout* vlyt = new QVBoxLayout;
        vlyt->addWidget(new QLabel("Line Chart Properties", this));
        section->setContentLayout(*vlyt);
    }

    // create graph on m_qcp and set graph index to chartInfo, and plot it
    chartInfo.SetQCustomPlot(m_qcp);
    chartInfo.SetGraphIndex(m_qcp->graphCount());
    m_qcp->addGraph();

    int graphIndex;
    graphIndex = chartInfo.GetGraphIndex();

    return section;
}

void lcChartWizard::Layouts()
{
    QSplitter* splt = new QSplitter(Qt::Horizontal, this);

    QVBoxLayout* vlytLeft = new QVBoxLayout;
    vlytLeft->setContentsMargins(0, 0, 0, 0);
    {
        vlytLeft->addStretch(1); // Add stretch to fill remaining space
    }

    QWidget* widLeft = new QWidget(this);
    widLeft->setLayout(vlytLeft);


    QVBoxLayout* vlytRight = new QVBoxLayout;
    vlytRight->setContentsMargins(0, 0, 0, 0);
    {
        m_qcp = new QCustomPlot(this);
        vlytRight->addWidget(m_qcp);
    }
    QWidget* widRight = new QWidget(this);
    widRight->setLayout(vlytRight);
    
    splt->addWidget(widLeft);
    splt->addWidget(widRight);
    splt->setSizes({200, 400}); // Set initial sizes for the splitter

    QVBoxLayout* vlytMain = new QVBoxLayout(this);
    vlytMain->setContentsMargins(0, 0, 0, 0);
    vlytMain->addWidget(splt);
    setLayout(vlytMain);
}

void lcChartWizard::UI()
{
    Widgets();
    Layouts();
}