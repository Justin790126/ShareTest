#include "ModelCellProfile.h"

ModelCellProfile::ModelCellProfile(QObject *parent) : QAbstractItemModel(parent) {
    m_rootItem = new CellEntry("Root", QRectF(), 0);
}

ModelCellProfile::~ModelCellProfile() { delete m_rootItem; }

void ModelCellProfile::loadFakeData(int count) {
    beginResetModel();
    CellEntry* topCell = new CellEntry("TOP_CELL", QRectF(0,0,5000,5000), 0, m_rootItem);
    m_rootItem->addChild(topCell);

    for(int i = 0; i < count; ++i) {
        CellEntry* child = new CellEntry(QString("Cell_%1").arg(i), 
                                         QRectF(i*10, i*10, 100, 100), i*1024, topCell);
        topCell->addChild(child);
    }
    endResetModel();
}

QModelIndex ModelCellProfile::index(int row, int column, const QModelIndex &parent) const {
    CellEntry *parentItem = itemFromIndex(parent);
    CellEntry *childItem = parentItem->children().value(row);
    return childItem ? createIndex(row, column, childItem) : QModelIndex();
}

QModelIndex ModelCellProfile::parent(const QModelIndex &child) const {
    if (!child.isValid()) return QModelIndex();
    CellEntry *childItem = itemFromIndex(child);
    CellEntry *parentItem = childItem->parent();
    if (parentItem == m_rootItem) return QModelIndex();
    return createIndex(parentItem->row(), 0, parentItem);
}

int ModelCellProfile::rowCount(const QModelIndex &parent) const {
    return itemFromIndex(parent)->children().count();
}

QVariant ModelCellProfile::data(const QModelIndex &index, int role) const {
    if (!index.isValid()) return QVariant();
    if (role == Qt::DisplayRole) return itemFromIndex(index)->name();
    return QVariant();
}

CellEntry* ModelCellProfile::itemFromIndex(const QModelIndex &index) const {
    if (index.isValid()) return static_cast<CellEntry*>(index.internalPointer());
    return m_rootItem;
}