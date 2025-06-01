#include "ViewLineChartProps.h"

ViewLineChartProps::ViewLineChartProps(const QString& title, const int animationDuration, QWidget* parent)
    : PropsSection(title, animationDuration, parent)
{
    UI();
}

void ViewLineChartProps::UI()
{
    QVBoxLayout *vlyt = new QVBoxLayout;
    {
        QHBoxLayout* hlytShowGraph = new QHBoxLayout;
        {
            chbShowGraph = new QCheckBox(tr("Show Graph"));
            chbShowGraph->setChecked(true); // Default to checked
            hlytShowGraph->addWidget(chbShowGraph);
        }

        QHBoxLayout* hlytLineName = new QHBoxLayout;
        {
            QLabel* lblLineName = new QLabel(tr("Line Name:"));
            hlytLineName->addWidget(lblLineName);
            leLineName = new QLineEdit;
            leLineName->setPlaceholderText(tr("Enter line name"));
            hlytLineName->addWidget(leLineName);
        }

        QHBoxLayout* hlytDotStyle = new QHBoxLayout;
        {
            QLabel* lblDotStyle = new QLabel(tr("Dot Style:"));
            hlytDotStyle->addWidget(lblDotStyle);
            cbbDotStyle = new QComboBox;
            // add scatter style that qcustomplot2.1 support in combobox
            cbbDotStyle->addItem(tr("None"), static_cast<int>(QCPScatterStyle::ssNone));
            // cbbDotStyle->addItem(tr("Dot"), static_cast<int>(QCPScatterStyle::ssDot));
            cbbDotStyle->addItem(tr("Cross"), static_cast<int>(QCPScatterStyle::ssCross));
            cbbDotStyle->addItem(tr("Plus"), static_cast<int>(QCPScatterStyle::ssPlus));
            cbbDotStyle->addItem(tr("Circle"), static_cast<int>(QCPScatterStyle::ssCircle));
            cbbDotStyle->addItem(tr("Disc"), static_cast<int>(QCPScatterStyle::ssDisc));
            cbbDotStyle->addItem(tr("Square"), static_cast<int>(QCPScatterStyle::ssSquare));
            cbbDotStyle->addItem(tr("Diamond"), static_cast<int>(QCPScatterStyle::ssDiamond));
            cbbDotStyle->addItem(tr("Star"), static_cast<int>(QCPScatterStyle::ssStar));
            cbbDotStyle->addItem(tr("Triangle"), static_cast<int>(QCPScatterStyle::ssTriangle));
            cbbDotStyle->addItem(tr("TriangleInverted"), static_cast<int>(QCPScatterStyle::ssTriangleInverted));
            cbbDotStyle->addItem(tr("CrossSquare"), static_cast<int>(QCPScatterStyle::ssCrossSquare));
            cbbDotStyle->addItem(tr("PlusSquare"), static_cast<int>(QCPScatterStyle::ssPlusSquare));
            cbbDotStyle->addItem(tr("CrossCircle"), static_cast<int>(QCPScatterStyle::ssCrossCircle));
            cbbDotStyle->addItem(tr("PlusCircle"), static_cast<int>(QCPScatterStyle::ssPlusCircle));
            cbbDotStyle->addItem(tr("Peace"), static_cast<int>(QCPScatterStyle::ssPeace));

            
            cbbDotStyle->setCurrentIndex(1); // Set default to None
            hlytDotStyle->addWidget(cbbDotStyle);   
        }

        QHBoxLayout* hlytDotWidth = new QHBoxLayout;
        {
            QLabel* lblDotSize = new QLabel(tr("Dot Size:"));
            hlytDotWidth->addWidget(lblDotSize);
            dsbDotSize = new QDoubleSpinBox;
            dsbDotSize->setRange(1, 20.0);
            dsbDotSize->setSingleStep(1);
            dsbDotSize->setValue(5.0); // Default dot size
            hlytDotWidth->addWidget(dsbDotSize);
        }

        QHBoxLayout* hlytShowLineSegment = new QHBoxLayout;
        {
            chbShowLineSegment = new QCheckBox(tr("Show Line Segment"));
            chbShowLineSegment->setChecked(true); // Default to checked
            hlytShowLineSegment->addWidget(chbShowLineSegment);
        }

        QHBoxLayout* hlytLineWidth = new QHBoxLayout;
        {
            QLabel* lblLineWidth = new QLabel(tr("Line Width:"));
            hlytLineWidth->addWidget(lblLineWidth);
            dsbLineWidth = new QDoubleSpinBox;
            dsbLineWidth->setRange(1, 20.0);
            dsbLineWidth->setSingleStep(1);
            dsbLineWidth->setValue(1.0); // Default line width
            hlytLineWidth->addWidget(dsbLineWidth);
        }

        QHBoxLayout* hlytLineColor = new QHBoxLayout;
        {
            QLabel* lblLineColor = new QLabel(tr("Line Color:"));
            hlytLineColor->addWidget(lblLineColor);
            leLineColor = new QLineEdit;
            leLineColor->setPlaceholderText(tr("Enter line color (e.g., #FF0000)"));
            hlytLineColor->addWidget(leLineColor);
        }

        QHBoxLayout* hlytThresAndMetrology = new QHBoxLayout;
        {
            chbShowThresholdAndMetrology = new QCheckBox(tr("Show Threshold and Metrology"));
            chbShowThresholdAndMetrology->setChecked(true); // Default to unchecked
            hlytThresAndMetrology->addWidget(chbShowThresholdAndMetrology);
        }

        vlyt->addLayout(hlytShowGraph);
        vlyt->addLayout(hlytLineName);
        vlyt->addLayout(hlytDotStyle);
        vlyt->addLayout(hlytDotWidth);
        vlyt->addLayout(hlytShowLineSegment);
        vlyt->addLayout(hlytLineWidth);
        vlyt->addLayout(hlytLineColor);
        vlyt->addLayout(hlytThresAndMetrology);

        connect(chbShowGraph, SIGNAL(toggled(bool)),
                this, SIGNAL(showGraphChanged(bool)));
        connect(leLineName, SIGNAL(textChanged(const QString&)),
                this, SIGNAL(lineNameChanged(const QString&)));
        connect(cbbDotStyle, SIGNAL(currentIndexChanged(int)),
                this, SIGNAL(dotStyleChanged(int)));
        connect(dsbDotSize, SIGNAL(valueChanged(double)),
                this, SIGNAL(dotSizeChanged(double)));
        connect(dsbLineWidth, SIGNAL(valueChanged(double)),
                this, SIGNAL(lineWidthChanged(double)));
        connect(leLineColor, SIGNAL(textChanged(const QString&)),
                this, SIGNAL(lineColorChanged(const QString&)));
        connect(chbShowLineSegment, SIGNAL(toggled(bool)),
                this, SIGNAL(showLineSegmentChanged(bool)));
        connect(chbShowThresholdAndMetrology, SIGNAL(toggled(bool)),
                this, SIGNAL(showThresholdAndMetrologyChanged(bool)));
    }
    this->setContentLayout(*vlyt);
}

ViewLineChartProps::~ViewLineChartProps()
{
    // Destructor implementation
}