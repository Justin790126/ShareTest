#include <QApplication>
#include <QWidget>
#include <QPainter>
#include <QPixmap>
#include <QPen>
#include <QtGui>
#include <QtCore>

#include "MyWidget.h"

class MainWidget : public QWidget {
    Q_OBJECT
public:
    MainWidget(QWidget *parent = NULL) : QWidget(parent) {
        // Initialize cache
        QVBoxLayout* lytmain = new QVBoxLayout;

        paintWid = new MyWidget;
        paintWid->resize(640,440);

        QHBoxLayout* lytBtns = new QHBoxLayout;
        {
            btnCache = new QPushButton("Render to Cache");
            btnRender = new QPushButton("Render");
            btnClear = new QPushButton("Clear");
            btnRenderPartial = new QPushButton("Render partial cache");
            lytBtns->addStretch();
            lytBtns->addWidget(btnCache);
            lytBtns->addWidget(btnRender);
            lytBtns->addWidget(btnClear);
            lytBtns->addWidget(btnRenderPartial);
        }

        lytmain->addWidget(paintWid);
        lytmain->addLayout(lytBtns);
        
        setLayout(lytmain);


        connect(btnCache, SIGNAL(clicked()), paintWid, SLOT(renderToCache()));
        connect(btnRender, SIGNAL(clicked()), paintWid, SLOT(render()));
        connect(btnClear, SIGNAL(clicked()), paintWid, SLOT(clearCache()));
        connect(btnRenderPartial, SIGNAL(clicked()), paintWid, SLOT(renderPartial()));
    }

    QPushButton* btnCache;
    QPushButton* btnRender;
    QPushButton* btnClear;
    QPushButton* btnRenderPartial;
    MyWidget* paintWid;
};