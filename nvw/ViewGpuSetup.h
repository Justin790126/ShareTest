#ifndef VIEW_GPU_SETUP_H
#define VIEW_GPU_SETUP_H

#include <QWidget>
#include <QHBoxLayout>
#include <QTreeWidget>
#include <QStackedWidget>

class ViewGpuSetup : public QWidget
{
    Q_OBJECT

public:
    explicit ViewGpuSetup(QWidget *parent = 0);
    ~ViewGpuSetup();

private:
    QHBoxLayout *mainLayout;
    QTreeWidget *treeWidget;
    QStackedWidget *stackedWidget;

    void setupUi();
};

#endif // VIEW_GPU_SETUP_H