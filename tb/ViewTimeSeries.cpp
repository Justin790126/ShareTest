#include "ViewTimeSeries.h"

ViewTimeSeries::ViewTimeSeries(QWidget *parent) : QWidget(parent)
{
    // Initialize UI components, set up layouts, etc.
    UI();
    Connect();
}

void ViewTimeSeries::widgets()
{
    // Create UI components for time series plot, controls, etc.
}

void ViewTimeSeries::Connect()
{
    connect(btnShowAllTags, SIGNAL(clicked()), this, SLOT(handleAddChartOnLyt()));
}
#include <iostream>
using namespace std;
void ViewTimeSeries::handleAddChartOnLyt()
{
    PropsSection *section = new PropsSection;
    {
        QVBoxLayout *vlyt = new QVBoxLayout;
        QWidget* wid = new QWidget;
        wid->setStyleSheet("Background: #FFFFFF;");
        wid->setFixedHeight(250);
        vlyt->addWidget(wid);

        QCustomPlot *plot = new QCustomPlot(wid);
        plot->resize(wid->size());

        section->setContentLayout(*vlyt);
    }
    vlytCharts->addWidget(section);
}

QFrame *ViewTimeSeries::CreateVerticalSeparator()
{
    // use QFrame to create vertical separator
    QFrame *sep = new QFrame(this);
    sep->setFrameShape(QFrame::VLine);
    sep->setFrameShadow(QFrame::Sunken);
    return sep;
}

QWidget *ViewTimeSeries::CreatePlotWidget()
{
    btnFilterTags = new QPushButton("Filter");
    leFilterTags = new QLineEdit;
    leFilterTags->setPlaceholderText(tr("Filter tags(regex)"));

    btnShowAllTags = new QPushButton(tr("All"));
    btnShowAllTags->setCheckable(true);
    btnShowScalars = new QPushButton(tr("Scalars"));
    btnShowScalars->setCheckable(true);
    btnShowImages = new QPushButton(tr("Images"));
    btnShowImages->setCheckable(true);
    btnShowHistograms = new QPushButton(tr("Histograms"));
    btnShowHistograms->setCheckable(true);
    bgShowCharts = new QButtonGroup;
    bgShowCharts->setExclusive(true);
    // bgShowCharts->addButton(btnShowAllTags, 0);
    bgShowCharts->addButton(btnShowScalars, 1);
    bgShowCharts->addButton(btnShowImages, 2);
    bgShowCharts->addButton(btnShowHistograms, 3);

    btnSettings = new QPushButton(tr("Settings"));
    btnSettings->setCheckable(true);

    QWidget *widSetting = CreateChartSettingsWidget();

    QWidget *wid = new QWidget;
    {
        QVBoxLayout *vlytPlot = new QVBoxLayout;
        {
            QHBoxLayout *hlytToolBar = new QHBoxLayout;
            {
                hlytToolBar->addWidget(btnFilterTags);
                hlytToolBar->addWidget(leFilterTags);
                hlytToolBar->addWidget(btnShowAllTags);
                hlytToolBar->addWidget(btnShowScalars);
                hlytToolBar->addWidget(btnShowImages);
                hlytToolBar->addWidget(btnShowHistograms);
                hlytToolBar->addWidget(CreateVerticalSeparator());
                hlytToolBar->addWidget(btnSettings);
            }

            QWidget *widChartContent = new QWidget;
            {
                QHBoxLayout *hlytChartContent = new QHBoxLayout;
                {
                    widCharts = new QWidget;
                    vlytCharts = new QVBoxLayout(widCharts);
                    {
                    }

                    QScrollArea* scaCharts = new QScrollArea;
                    scaCharts->setWidget(widCharts);
                    scaCharts->setWidgetResizable(true);

                    hlytChartContent->addWidget(scaCharts, 7);
                    hlytChartContent->addWidget(widSetting, 3);
                }
                widChartContent->setLayout(hlytChartContent);
            }

            vlytPlot->addLayout(hlytToolBar);
            vlytPlot->addWidget(widChartContent);
        }
        wid->setLayout(vlytPlot);
    }
    return wid;
}

QWidget *ViewTimeSeries::CreateJobVisualWidget()
{
    twJobFiles = new QTreeWidget;
    twJobFiles->setColumnCount(2);
    btnFilterJobFiles = new QPushButton(tr("Filter"));
    leFilterJobFiles = new QLineEdit;
    leFilterJobFiles->setPlaceholderText(tr("Filter runs(regex)"));

    QWidget *widChartVis = new QWidget;
    {
        QVBoxLayout *vlytJobs = new QVBoxLayout;
        {
            QHBoxLayout *hlytFilter = new QHBoxLayout;
            {
                hlytFilter->addWidget(btnFilterJobFiles);
                hlytFilter->addWidget(leFilterJobFiles);
            }
            vlytJobs->addLayout(hlytFilter);
            vlytJobs->addWidget(twJobFiles);
        }
        widChartVis->setLayout(vlytJobs);
    }
    return widChartVis;
}

QWidget *ViewTimeSeries::CreateChartSettingsWidget()
{
    QWidget *wid = new QWidget;
    {
        QVBoxLayout *vlytChartSetting = new QVBoxLayout;
        {
            // add group box with bold name "GENERAL"
            QGroupBox *grpGeneral = new QGroupBox(tr("GENERAL"));
            {
                // add settings for general chart settings
                //...
            }
            vlytChartSetting->addWidget(grpGeneral);

            // add group box with bold name "SCALARS"
            QGroupBox *grpScalars = new QGroupBox(tr("SCALARS"));
            {
                // add settings for scalar chart settings
                //...
            }
            vlytChartSetting->addWidget(grpScalars);

            // add group box with bold name "IMAGES"
            QGroupBox *grpImages = new QGroupBox(tr("IMAGES"));
            {
                // add settings for image chart settings
                //...
            }
            vlytChartSetting->addWidget(grpImages);

            // add group box with bold name "HISTOGRAMS"
            QGroupBox *grpHistograms = new QGroupBox(tr("HISTOGRAMS"));
            {
                // add settings for histogram chart settings
                //...
            }
            vlytChartSetting->addWidget(grpHistograms);
        }
        wid->setLayout(vlytChartSetting);
    }
    return wid;
}

void ViewTimeSeries::layouts()
{
    // Set up layouts to arrange UI components
    QHBoxLayout *hlytMain = new QHBoxLayout;
    hlytMain->setContentsMargins(0, 0, 0, 0);
    {
        QSplitter *spltRunAndPlot = new QSplitter;
        {
            QWidget *widChartVis = CreateJobVisualWidget();
            QWidget *widPlot = CreatePlotWidget();
            spltRunAndPlot->addWidget(widChartVis);
            spltRunAndPlot->addWidget(widPlot);
        }
        // set ratio of spltRunAndPlot to 2:8
        spltRunAndPlot->setStretchFactor(0, 2);
        spltRunAndPlot->setStretchFactor(1, 8);
        hlytMain->addWidget(spltRunAndPlot);
    }
    this->setLayout(hlytMain);
}

void ViewTimeSeries::UI()
{
    widgets();
    layouts();

    // Set up the window
    setWindowTitle(tr("IHT Time Series"));
    setMinimumSize(800, 600);
}

// Example implementation of a simple time series plot using Qt Charts