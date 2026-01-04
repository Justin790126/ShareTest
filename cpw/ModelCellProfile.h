#ifndef MODELCELLPROFILE_H
#define MODELCELLPROFILE_H

#include <QAbstractItemModel>
#include "CellEntry.h"

class ModelCellProfile : public QAbstractItemModel {
    Q_OBJECT
public:
    ModelCellProfile(QObject *parent = 0);
    ~ModelCellProfile();

    QModelIndex index(int row, int column, const QModelIndex &parent) const;
    QModelIndex parent(const QModelIndex &child) const;
    int rowCount(const QModelIndex &parent) const;
    int columnCount(const QModelIndex &parent) const { return 1; }
    QVariant data(const QModelIndex &index, int role) const;

    void loadFakeData(int count);
    CellEntry* itemFromIndex(const QModelIndex &index) const;

private:
    CellEntry* m_rootItem;
};
#endif