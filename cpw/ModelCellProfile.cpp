#include "ModelCellProfile.h"
#include <QFile>
#include <QDataStream>

ModelCellProfile::ModelCellProfile(QObject *parent) : QAbstractItemModel(parent) {
    m_rootItem = new CellEntry("InternalRoot", QRectF());
}

ModelCellProfile::~ModelCellProfile() { delete m_rootItem; }

void ModelCellProfile::startLoading(const QString &fileName) {
    ParseWorker *worker = new ParseWorker(fileName, m_rootItem);
    connect(worker, SIGNAL(finished()), this, SLOT(handleWorkerFinished()));
    connect(worker, SIGNAL(finished()), worker, SLOT(deleteLater()));
    worker->start();
}

void ModelCellProfile::handleWorkerFinished() {
    beginResetModel();
    // 這裡不需要額外邏輯，因為 Worker 已經把數據填入 m_rootItem
    endResetModel();
    emit loadingFinished();
}
void ParseWorker::run() {
    QFile file(m_file);
    if (!file.open(QIODevice::ReadOnly)) return;

    QDataStream in(&file);
    in.setByteOrder(QDataStream::LittleEndian);

    // 1. 讀取 Header
    unsigned int vlen; in >> vlen;
    file.read(vlen); 
    unsigned int pid; in >> pid;

    CellEntry* topCell = NULL;

    // 2. 讀取所有 Cells
    while (!file.atEnd()) {
        unsigned int ref, nlen;
        in >> ref >> nlen;
        if (in.status() != QDataStream::Ok) break;

        QString name = QString::fromUtf8(file.read(nlen));
        double l, b, r, t;
        in >> l >> b >> r >> t;
        
        // Skip 25 bytes (pos, findex, count, flag)
        unsigned long long d64; unsigned char d8;
        in >> d64 >> d64 >> d64 >> d8;

        if (!topCell) {
            // 將檔案中的第一筆資料作為頂層物件，掛在 m_root 之下
            topCell = new CellEntry(name, QRectF(l, -t, r-l, t-b), m_root);
            m_root->addChild(topCell);
        } else {
            // 後續所有資料都掛在剛才建立的 topCell 之下
            topCell->addChild(new CellEntry(name, QRectF(l, -t, r-l, t-b), topCell));
        }
    }
    
    file.close();
    emit finished();
}
// ... 保持原本的 index, parent, rowCount, data 實作 ...
QModelIndex ModelCellProfile::index(int row, int column, const QModelIndex &parent) const {
    CellEntry *p = itemFromIndex(parent);
    CellEntry *c = p->children().value(row);
    return c ? createIndex(row, column, c) : QModelIndex();
}

QModelIndex ModelCellProfile::parent(const QModelIndex &child) const {
    if (!child.isValid()) return QModelIndex();
    CellEntry *c = itemFromIndex(child);
    CellEntry *p = c->parent();
    if (p == m_rootItem) return QModelIndex();
    return createIndex(p->row(), 0, p);
}

int ModelCellProfile::rowCount(const QModelIndex &parent) const {
    return itemFromIndex(parent)->children().count();
}

QVariant ModelCellProfile::data(const QModelIndex &index, int role) const {
    if (!index.isValid() || role != Qt::DisplayRole) return QVariant();
    return itemFromIndex(index)->name();
}

CellEntry* ModelCellProfile::itemFromIndex(const QModelIndex &index) const {
    if (index.isValid()) return static_cast<CellEntry*>(index.internalPointer());
    return m_rootItem;
}