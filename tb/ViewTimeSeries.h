#ifndef VIEW_TIME_SERIES
#define VIEW_TIME_SERIES

#include <QtGui>
#include "PropsSection.h"
#include "qcustomplot.h"


class ViewTimeSeries : public QWidget
{
    Q_OBJECT
    public:
        ViewTimeSeries(QWidget *parent=NULL);
        ~ViewTimeSeries() = default;

        void widgets();
        void layouts();
        void UI();
        void Connect();

        QWidget* AddChartSection(QString title="");

        QTreeWidget* GetTwJobsTree() { return twJobFiles; }
    
    private:
        QTreeWidget* twJobFiles;
        QLineEdit* leFilterJobFiles;
        QPushButton* btnFilterJobFiles;
        QWidget* CreateJobVisualWidget();
        QPushButton* btnFilterTags;
        QLineEdit* leFilterTags;
        QButtonGroup* bgShowCharts;
        QPushButton* btnShowAllTags;
        QPushButton* btnShowScalars;
        QPushButton* btnShowImages;
        QPushButton* btnShowHistograms;
        QPushButton* btnSettings;
        QWidget* widCharts;
        QVBoxLayout* vlytCharts;
        QWidget* CreatePlotWidget();
        QFrame* CreateVerticalSeparator();
        QWidget* widSettings;
        QWidget* CreateChartSettingsWidget();
    
    private slots:
        void handleAddChartOnLyt();
        void handleSettingsOnOff();
};


#endif /* VIEW_TIME_SERIES */