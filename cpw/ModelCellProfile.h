#ifndef MODELCELLPROFILE_H
#define MODELCELLPROFILE_H
#include <QAbstractItemModel>
#include <QThread>
#include "CellEntry.h"

class ParseWorker : public QThread {
    Q_OBJECT
public:
    ParseWorker(const QString &file, CellEntry* topGrp, CellEntry* allGrp) 
        : m_file(file), m_groupTop(topGrp), m_groupAll(allGrp) {}
    void run();

signals:
    void finished();

public: // 確保變數在此公開，供 run() 存取
    QString m_file;
    CellEntry* m_groupTop;
    CellEntry* m_groupAll;
};

class ModelCellProfile : public QAbstractItemModel {
    Q_OBJECT
public:
    explicit ModelCellProfile(QObject *parent = 0);
    ~ModelCellProfile();
    QModelIndex index(int row, int column, const QModelIndex &parent) const;
    QModelIndex parent(const QModelIndex &child) const;
    int rowCount(const QModelIndex &parent) const;
    int columnCount(const QModelIndex &parent) const { return 1; }
    QVariant data(const QModelIndex &index, int role) const;
    void startLoading(const QString &fileName);
    CellEntry* itemFromIndex(const QModelIndex &index) const;

private slots:
    void onWorkerFinished();

signals:
    void loadingFinished();

private:
    CellEntry* m_rootItem;
    CellEntry* m_groupTop;
    CellEntry* m_groupAll;
};
#endif