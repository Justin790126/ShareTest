#include <QtGui>
#include <QApplication>
#include <QTableView>
#include <QAbstractTableModel>
#include <QTimer>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QSplitter>
#include <QLineEdit>
#include <QCheckBox>
#include <QFile>
#include <QTextStream>
#include <QLabel>
#include <QGroupBox>
#include "qcustomplot.h"
#include <dirent.h>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>

// Model: Manages process data with CPU and memory history
class ProcessModel : public QAbstractTableModel {
    Q_OBJECT
public:
    ProcessModel(QObject* parent = 0)
        : QAbstractTableModel(parent), selectedPid(-1), recordToFile(false),
          maxCpuUsage(0.0), maxMemoryUsage(0) {}

    struct UsageData {
        double cpuUsage;
        long memoryUsage;
    };

    int rowCount(const QModelIndex& parent = QModelIndex()) const {
        return processes.size();
    }

    int columnCount(const QModelIndex& parent = QModelIndex()) const {
        return 5; // PID, Username, Process Name, CPU Usage, Memory Usage
    }

    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const {
        if (!index.isValid() || role != Qt::DisplayRole) return QVariant();

        const ProcessInfo& info = processes[index.row()];
        switch (index.column()) {
            case 0: return info.pid;
            case 1: return info.username;
            case 2: return info.name;
            case 3: return QString::number(info.cpuUsage, 'f', 2);
            case 4: return QVariant(static_cast<qlonglong>(info.memoryUsage));
            default: return QVariant();
        }
    }

    QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const {
        if (role != Qt::DisplayRole || orientation != Qt::Horizontal) return QVariant();
        switch (section) {
            case 0: return "PID";
            case 1: return "Username";
            case 2: return "Process Name";
            case 3: return "CPU Usage (%)";
            case 4: return "Memory Usage (KB)";
            default: return QVariant();
        }
    }

    void updateProcesses(const QString& nameFilter, const QString& pidFilter, const QString& userFilter) {
        beginResetModel();
        processes.clear();

        // Update selectedPid based on pidFilter
        bool ok;
        pid_t newPid = pidFilter.toInt(&ok);
        if (ok && newPid != selectedPid) {
            selectedPid = newPid;
            usageHistory.clear(); // Clear history when PID changes
            maxCpuUsage = 0.0;    // Reset max values when PID changes
            maxMemoryUsage = 0;
        } else if (!ok) {
            selectedPid = -1; // Invalid PID filter clears selection
        }

        DIR* dir = opendir("/proc");
        if (!dir) {
            endResetModel();
            return;
        }

        struct dirent* entry;
        while ((entry = readdir(dir)) != NULL) {
            pid_t pid = atoi(entry->d_name);
            if (pid == 0) continue;

            QString username = getUsername(pid);
            QString name = getProcessName(pid);

            // Apply filters
            bool userMatch = userFilter.isEmpty() || username.toLower().contains(userFilter.toLower());
            bool nameMatch = nameFilter.isEmpty() || name.toLower().contains(nameFilter.toLower());
            bool pidMatch = pidFilter.isEmpty() || QString::number(pid).contains(pidFilter);

            if (userMatch && nameMatch && pidMatch) {
                ProcessInfo info;
                info.pid = pid;
                info.username = username;
                info.name = name;
                info.cpuUsage = getCpuUsage(pid);
                info.memoryUsage = getMemoryUsage(pid);
                processes.append(info);

                // Update usage history and max values for selected PID
                if (pid == selectedPid) {
                    qint64 time = QDateTime::currentMSecsSinceEpoch();
                    UsageData usage;
                    usage.cpuUsage = info.cpuUsage;
                    usage.memoryUsage = info.memoryUsage;
                    usageHistory[time / 1000.0] = usage;

                    // Update maximum values
                    maxCpuUsage = qMax(maxCpuUsage, info.cpuUsage);
                    maxMemoryUsage = qMax(maxMemoryUsage, info.memoryUsage);

                    if (recordToFile) {
                        QString filename = QString("%1_%2_%3_usage.txt")
                                              .arg(username)
                                              .arg(name)
                                              .arg(pid);
                        QFile file(filename);
                        bool isNewFile = !file.exists();
                        if (file.open(QIODevice::Append | QIODevice::Text)) {
                            QTextStream out(&file);
                            if (isNewFile) {
                                out << "timestamp(ms),CPU(%),Memory(KB)\n";
                            }
                            out << time << "," << QString::number(info.cpuUsage, 'f', 2) << "," << info.memoryUsage << "\n";
                            file.close();
                        }
                    }
                }
            }
        }
        closedir(dir);
        endResetModel();
    }

    void setRecordToFile(bool record) {
        recordToFile = record;
    }

    QMap<double, UsageData> getUsageHistory() const { return usageHistory; }
    double getMaxCpuUsage() const { return maxCpuUsage; }
    long getMaxMemoryUsage() const { return maxMemoryUsage; }

private:
    struct ProcessInfo {
        pid_t pid;
        QString username;
        QString name;
        double cpuUsage;
        long memoryUsage;
    };

    QList<ProcessInfo> processes;
    QMap<pid_t, long> prevCpuTimes;
    pid_t selectedPid;
    QMap<double, UsageData> usageHistory;
    bool recordToFile;
    double maxCpuUsage;    // Maximum CPU usage for selected PID
    long maxMemoryUsage;   // Maximum memory usage for selected PID

    QString getUsername(pid_t pid) {
        std::ostringstream path;
        path << "/proc/" << pid << "/status";
        std::ifstream file(path.str().c_str());
        if (!file.is_open()) return "N/A";

        std::string line;
        while (std::getline(file, line)) {
            if (line.find("Uid:") == 0) {
                std::istringstream iss(line);
                std::string uidLabel;
                uid_t uid;
                iss >> uidLabel >> uid;
                struct passwd* pw = getpwuid(uid);
                return pw ? QString(pw->pw_name) : "Unknown";
            }
        }
        return "N/A";
    }

    QString getProcessName(pid_t pid) {
        std::ostringstream path;
        path << "/proc/" << pid << "/stat";
        std::ifstream file(path.str().c_str());
        if (!file.is_open()) return "N/A";

        std::string line;
        std::getline(file, line);
        std::istringstream iss(line);

        std::string pidStr, comm;
        iss >> pidStr >> comm;
        if (comm.size() > 2) {
            return QString::fromStdString(comm.substr(1, comm.size() - 2));
        }
        return "N/A";
    }

    double getCpuUsage(pid_t pid) {
        std::ostringstream path;
        path << "/proc/" << pid << "/stat";
        std::ifstream file(path.str().c_str());
        if (!file.is_open()) return 0.0;

        std::string line;
        std::getline(file, line);
        std::istringstream iss(line);

        std::string pidStr, comm;
        long utime, stime;
        iss >> pidStr >> comm;
        for (int i = 2; i < 14; ++i) iss >> comm;
        iss >> utime >> stime;

        long totalTime = utime + stime;
        double cpuUsage = 0.0;

        if (prevCpuTimes.contains(pid)) {
            long prevTime = prevCpuTimes[pid];
            double elapsed = 1.0;
            cpuUsage = 100.0 * (totalTime - prevTime) / (elapsed * sysconf(_SC_CLK_TCK));
        }
        prevCpuTimes[pid] = totalTime;

        return cpuUsage;
    }

    long getMemoryUsage(pid_t pid) {
        std::ostringstream path;
        path << "/proc/" << pid << "/status";
        std::ifstream file(path.str().c_str());
        if (!file.is_open()) return 0;

        std::string line;
        while (std::getline(file, line)) {
            if (line.find("VmRSS:") == 0) {
                std::istringstream iss(line);
                std::string vmRssLabel;
                long mem;
                std::string unit;
                iss >> vmRssLabel >> mem >> unit;
                return mem;
            }
        }
        return 0;
    }
};

// View: Displays table, plot, and checkbox with QSplitter
class ProcessView : public QWidget {
    Q_OBJECT
public:
    ProcessView(QWidget* parent = 0) : QWidget(parent) {
        // Filters
        QHBoxLayout* hlytFilterUser = new QHBoxLayout();
        userFilterEdit = new QLineEdit(this);
        userFilterEdit->setPlaceholderText("Filter by username...");
        hlytFilterUser->addWidget(new QLabel("Filter by Username: "));
        hlytFilterUser->addWidget(userFilterEdit);

        QHBoxLayout* hlytFilterName = new QHBoxLayout();
        nameFilterEdit = new QLineEdit(this);
        nameFilterEdit->setPlaceholderText("Find by process name (partial match)...");
        hlytFilterName->addWidget(new QLabel("Find by Process Name: "));
        hlytFilterName->addWidget(nameFilterEdit);

        QHBoxLayout* hlytFilterPid = new QHBoxLayout;
        pidFilterEdit = new QLineEdit(this);
        pidFilterEdit->setPlaceholderText("Filter by PID (plot this PID)...");
        hlytFilterPid->addWidget(new QLabel("Filter by PID: "));
        hlytFilterPid->addWidget(pidFilterEdit);

        // Table
        tableView = new QTableView(this);
        tableView->setSelectionBehavior(QAbstractItemView::SelectRows);
        tableView->setEditTriggers(QAbstractItemView::NoEditTriggers);

        // Checkbox
        recordCheckBox = new QCheckBox("Record to file", this);
        hlytFilterPid->addWidget(recordCheckBox);

        // Left layout (filters, table, checkbox)
        QVBoxLayout* leftLayout = new QVBoxLayout;
        leftLayout->addLayout(hlytFilterUser);
        leftLayout->addLayout(hlytFilterName);
        leftLayout->addLayout(hlytFilterPid);
        leftLayout->addWidget(tableView);

        // Plot
        plot = new QCustomPlot(this);
        plot->addGraph(); // Memory usage (red)
        plot->graph(0)->setPen(QPen(Qt::red));
        plot->xAxis->setLabel("Time (s)");
        plot->yAxis->setLabel("Memory Usage (KB)");
        plot->legend->setVisible(true);
        plot->graph(0)->setName("Memory");
        plot->setMinimumHeight(200);
        plot->axisRect()->insetLayout()->setInsetAlignment(0, Qt::AlignBottom | Qt::AlignRight);

        // Right layout with QGroupBox
        QWidget* rightWidget = new QWidget(this);
        
        QVBoxLayout* vlytRight = new QVBoxLayout;
        
        QGroupBox* gpbRtStatistics = new QGroupBox("Live Statistics", this);
        QVBoxLayout* statsLayout = new QVBoxLayout;
        gpbRtStatistics->setLayout(statsLayout);

        // Labels for max CPU and memory usage
        maxCpuLabel = new QLabel("Max CPU Usage: 0.00 %", this);
        maxMemoryLabel = new QLabel("Max Memory Usage: 0 KB", this);
        statsLayout->addWidget(maxCpuLabel);
        statsLayout->addWidget(maxMemoryLabel);
        statsLayout->addStretch(); // Push labels to the top

        vlytRight->addWidget(gpbRtStatistics);
        rightWidget->setLayout(vlytRight);


        // Splitter
        splitter = new QSplitter(Qt::Horizontal, this);
        QWidget* leftWidget = new QWidget(this);
        leftWidget->setLayout(leftLayout);
        splitter->addWidget(leftWidget);
        splitter->addWidget(plot);
        splitter->addWidget(rightWidget);
        QList<int> sizes = {400, 700, 200};
        splitter->setSizes(sizes);

        // Main layout
        QVBoxLayout* mainLayout = new QVBoxLayout(this);
        mainLayout->addWidget(splitter);

        QStatusBar* statusBar = new QStatusBar;
        statusBar->setFixedHeight(40);
        mainLayout->addWidget(statusBar);
        statusBar->showMessage("Ready");
    }

    void setModel(ProcessModel* model) {
        tableView->setModel(model);
    }

    QLineEdit* getUserFilterEdit() const { return userFilterEdit; }
    QLineEdit* getNameFilterEdit() const { return nameFilterEdit; }
    QLineEdit* getPidFilterEdit() const { return pidFilterEdit; }
    QTableView* getTableView() const { return tableView; }
    QCustomPlot* getPlot() const { return plot; }
    QCheckBox* getRecordCheckBox() const { return recordCheckBox; }
    QLabel* getMaxCpuLabel() const { return maxCpuLabel; }         // Getter for max CPU label
    QLabel* getMaxMemoryLabel() const { return maxMemoryLabel; }   // Getter for max memory label

private:
    QTableView* tableView;
    QLineEdit* userFilterEdit;
    QLineEdit* nameFilterEdit;
    QLineEdit* pidFilterEdit;
    QCustomPlot* plot;
    QCheckBox* recordCheckBox;
    QSplitter* splitter;
    QLabel* maxCpuLabel;      // Label for max CPU usage
    QLabel* maxMemoryLabel;   // Label for max memory usage
};

// Controller: Manages interactions, plotting, and recording
class ProcessController : public QObject {
    Q_OBJECT
public:
    ProcessController(ProcessModel* model, ProcessView* view, QObject* parent = 0)
        : QObject(parent), model(model), view(view) {
        view->setModel(model);

        connect(view->getUserFilterEdit(), SIGNAL(textChanged(const QString&)),
                this, SLOT(filtersChanged()));
        connect(view->getNameFilterEdit(), SIGNAL(textChanged(const QString&)),
                this, SLOT(filtersChanged()));
        connect(view->getPidFilterEdit(), SIGNAL(textChanged(const QString&)),
                this, SLOT(filtersChanged()));
        connect(view->getRecordCheckBox(), SIGNAL(toggled(bool)),
                this, SLOT(recordToggled(bool)));

        QTimer* timer = new QTimer(this);
        connect(timer, SIGNAL(timeout()), this, SLOT(update()));
        timer->start(1000);

        update(); // Initial update
    }

private slots:
    void filtersChanged() {
        update();
    }

    void recordToggled(bool checked) {
        model->setRecordToFile(checked);
    }

    void update() {
        model->updateProcesses(view->getNameFilterEdit()->text(),
                              view->getPidFilterEdit()->text(),
                              view->getUserFilterEdit()->text());
        updatePlot();
        updateStatistics(); // Update live statistics
    }

    void updatePlot() {
        QCustomPlot* plot = view->getPlot();
        QMap<double, ProcessModel::UsageData> usageData = model->getUsageHistory();

        QVector<double> time(usageData.keys().size()), mem(usageData.values().size());
        int i = 0;
        double minTime = usageData.isEmpty() ? 0 : usageData.keys().first();
        for (double t : usageData.keys()) {
            time[i] = t - minTime; // Relative time in seconds
            mem[i] = usageData[t].memoryUsage;
            i++;
        }

        plot->graph(0)->setData(time, mem);
        plot->xAxis->setRange(0, time.isEmpty() ? 10 : time.last() + 1);
        plot->yAxis->setRange(0, mem.isEmpty() ? 1000 : *std::max_element(mem.begin(), mem.end()) * 1.1);
        plot->replot();
    }

    void updateStatistics() {
        // Update the labels with max CPU and memory usage
        double maxCpu = model->getMaxCpuUsage();
        long maxMemory = model->getMaxMemoryUsage();
        view->getMaxCpuLabel()->setText(QString("Max CPU Usage: %1 %").arg(maxCpu, 0, 'f', 2));
        view->getMaxMemoryLabel()->setText(QString("Max Memory Usage: %1 KB").arg(maxMemory));
    }

private:
    ProcessModel* model;
    ProcessView* view;
};
