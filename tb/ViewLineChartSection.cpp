#include "ViewLineChartSection.h"

ViewLineChartProps::~ViewLineChartProps()
{
    for (size_t i = 0; i < m_vcInfos.size(); i++)
    {
        if (m_vcInfos[i])
            delete m_vcInfos[i];
    }
}

void ViewLineChartProps::DrawLineChart(ChartInfo *info)
{
    if (!m_qcp)
        return;
    if (!info)
        return;
    m_vcInfos.emplace_back(info);

    // use QCustomPlot to plot line chart
    // Clear any existing graphs
    // m_qcp->clearGraphs();

    // Create a graph and set data
    m_qcp->addGraph();
    int graphId = m_qcp->graphCount()-1;
    if (graphId < 0) return;
    m_qcp->graph(graphId)->setData(info->m_qvdXData, info->m_qvdYData);

    // Customize the graph appearance
    std::random_device rd;                       // Seed for randomness
    std::mt19937 gen(rd());                      // Mersenne Twister engine
    std::uniform_int_distribution<> dis(0, 255); // Range for RGB values (0-255)

    QColor randomColor(dis(gen), dis(gen), dis(gen)); // Random RGB
    m_qcp->graph(graphId)->setPen(QPen(randomColor)); // Line color
    m_qcp->graph(graphId)->setLineStyle(QCPGraph::lsLine);                                 // Solid line
    m_qcp->graph(graphId)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCircle, 5)); // Circular points

    // Set axis labels
    m_qcp->xAxis->setLabel(info->m_sXLabel.c_str());
    m_qcp->yAxis->setLabel(info->m_sYLabel.c_str());

    // Set axis ranges (adjust based on data)
    m_qcp->xAxis->setRange(0, info->m_qvdYData.size() - 1);
    float minY = *std::min_element(info->m_qvdYData.begin(), info->m_qvdYData.end());
    float maxY = *std::max_element(info->m_qvdYData.begin(), info->m_qvdYData.end());
    m_qcp->yAxis->setRange(minY - 0.1f * (maxY - minY), maxY + 0.1f * (maxY - minY)); // Add 10% padding

    // Add grid for better readability
    m_qcp->xAxis->setBasePen(QPen(Qt::black));
    m_qcp->yAxis->setBasePen(QPen(Qt::black));
    m_qcp->xAxis->grid()->setVisible(true);
    m_qcp->yAxis->grid()->setVisible(true);

    // Enable interactions (optional)
    m_qcp->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectPlottables);

    // Replot to display the graph
    m_qcp->replot();
}
