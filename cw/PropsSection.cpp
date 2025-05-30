/*
    Elypson/qt-collapsible-PropsSection
    (c) 2016 Michael A. Voelkel - michael.alexander.voelkel@gmail.com

    This file is part of Elypson/qt-collapsible PropsSection.

    Elypson/qt-collapsible-PropsSection is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Elypson/qt-collapsible-PropsSection is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Elypson/qt-collapsible-PropsSection. If not, see <http://www.gnu.org/licenses/>.
*/

#include <QPropertyAnimation>

#include "PropsSection.h"
#include <QDebug>

    PropsSection::PropsSection(const QString& title, const int animationDuration, QWidget* parent)
        : QWidget(parent), animationDuration(animationDuration)
    {
        toggleButton = new QToolButton(this);
        headerLine = new QFrame(this);
        toggleAnimation = new QParallelAnimationGroup(this);
        contentArea = new QScrollArea(this);
        mainLayout = new QGridLayout(this);
        btnClose = new QPushButton("x", this);
        btnClose->setFlat(true);

        toggleButton->setStyleSheet("QToolButton {border: none;}");
        toggleButton->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
        toggleButton->setArrowType(Qt::ArrowType::RightArrow);
        toggleButton->setText(title);
        toggleButton->setCheckable(true);
        toggleButton->setChecked(false);

        headerLine->setFrameShape(QFrame::HLine);
        headerLine->setFrameShadow(QFrame::Sunken);
        headerLine->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Maximum);

        contentArea->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);

        // start out collapsed
        contentArea->setMaximumHeight(0);
        contentArea->setMinimumHeight(0);

        // let the entire widget grow and shrink with its content
        toggleAnimation->addAnimation(new QPropertyAnimation(this, "maximumHeight"));
        toggleAnimation->addAnimation(new QPropertyAnimation(this, "minimumHeight"));
        toggleAnimation->addAnimation(new QPropertyAnimation(contentArea, "maximumHeight"));

        mainLayout->setVerticalSpacing(0);
        mainLayout->setContentsMargins(0, 0, 0, 0);

        int row = 0;    
        mainLayout->addWidget(toggleButton, row, 1, 1, 1, Qt::AlignLeft);
        mainLayout->addWidget(headerLine, row, 2, 1, 1);
        mainLayout->addWidget(btnClose, row++, 3, 1, 1, Qt::AlignRight);
        btnClose->setVisible(false);
        
        mainLayout->addWidget(contentArea, row, 0, 1, 3);
        setLayout(mainLayout);

        connect(toggleButton, SIGNAL(toggled(bool)), this, SLOT(handleToggle(bool)));
        connect(btnClose, SIGNAL(clicked()), this, SLOT(handleCloseButton()));
    }

    void PropsSection::handleCloseButton()
    {
        // delete this;
        emit PropsSectionClosed();
    }

    void PropsSection::handleToggle(bool expanded)
    {
        toggleButton->setArrowType(expanded ? Qt::ArrowType::DownArrow : Qt::ArrowType::RightArrow);
        toggleAnimation->setDirection(expanded ? QAbstractAnimation::Forward : QAbstractAnimation::Backward);
        toggleAnimation->start();
        this->isExpanded = expanded;
    }

    void PropsSection::setContentLayout(QLayout& contentLayout)
    {
        delete contentArea->layout();
        contentArea->setLayout(&contentLayout);
        collapsedHeight = sizeHint().height() - contentArea->maximumHeight();
        
        updateHeights();
    }
    
    void PropsSection::setTitle(QString title)
    {
        toggleButton->setText(std::move(title));
    }
    
    void PropsSection::updateHeights()
    {
        int contentHeight = contentArea->layout()->sizeHint().height();

        for (int i = 0; i < toggleAnimation->animationCount() - 1; ++i)
        {
            QPropertyAnimation* PropsSectionAnimation = static_cast<QPropertyAnimation *>(toggleAnimation->animationAt(i));
            PropsSectionAnimation->setDuration(animationDuration);
            PropsSectionAnimation->setStartValue(collapsedHeight);
            PropsSectionAnimation->setEndValue(collapsedHeight + contentHeight);
        }

        QPropertyAnimation* contentAnimation = static_cast<QPropertyAnimation *>(toggleAnimation->animationAt(toggleAnimation->animationCount() - 1));
        contentAnimation->setDuration(animationDuration);
        contentAnimation->setStartValue(0);
        contentAnimation->setEndValue(contentHeight);
        
        toggleAnimation->setDirection(isExpanded ? QAbstractAnimation::Forward : QAbstractAnimation::Backward);
        toggleAnimation->start();
    }
