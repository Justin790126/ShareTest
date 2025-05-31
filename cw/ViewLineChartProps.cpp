#include "ViewLineChartProps.h"

ViewLineChartProps::ViewLineChartProps(const QString& title, const int animationDuration, QWidget* parent)
    : ViewChartProps(title, animationDuration, parent)
{
    // Constructor implementation
    QVBoxLayout *vlyt = new QVBoxLayout;
    {
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
            cbbDotStyle->addItem(tr("Dot"), static_cast<int>(QCPScatterStyle::ssDot));
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

            
            cbbDotStyle->setCurrentIndex(0); // Set default to None
            hlytDotStyle->addWidget(cbbDotStyle);
            
        }

        vlyt->addLayout(hlytLineName);
        vlyt->addLayout(hlytDotStyle);

        connect(leLineName, SIGNAL(textChanged(const QString&)),
                this, SIGNAL(lineNameChanged(const QString&)));
        connect(cbbDotStyle, SIGNAL(currentIndexChanged(int)),
                this, SIGNAL(dotStyleChanged(int)));
    }
    this->setContentLayout(*vlyt);
}

ViewLineChartProps::~ViewLineChartProps()
{
    // Destructor implementation
}