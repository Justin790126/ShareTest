#ifndef TEST_WIDGET_H
#define TEST_WIDGET_H

#include <QtGui>

class TestWidget : public QWidget
{
    Q_OBJECT

public:
    TestWidget(QWidget *parent = 0);
    ~TestWidget() = default;

    void widgets();
    void layouts();
    void Connect();
    void UI();

    QPushButton* btnConnect;
    QPushButton* btnSend;
    QPushButton* btnRecv;
    QTextEdit* teOutput;
};

#endif /* TEST_WIDGET_H */