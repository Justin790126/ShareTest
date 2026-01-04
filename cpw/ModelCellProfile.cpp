#include "ModelCellProfile.h"
#include <QFile>
#include <QDataStream>

ModelCellProfile::ModelCellProfile(QObject *parent) : QAbstractItemModel(parent) {
    m_rootItem = new CellEntry("Root", QRectF());
    m_groupTop = new CellEntry("Top Cells", QRectF(), m_rootItem);
    m_groupAll = new CellEntry("All Cells", QRectF(), m_rootItem);
    m_rootItem->addChild(m_groupTop);
    m_rootItem->addChild(m_groupAll);
}
ModelCellProfile::~ModelCellProfile() { delete m_rootItem; }

void ModelCellProfile::startLoading(const QString &fileName) {
    qDeleteAll(m_groupTop->children()); m_groupTop->children().clear();
    qDeleteAll(m_groupAll->children()); m_groupAll->children().clear();
    ParseWorker *worker = new ParseWorker(fileName, m_groupTop, m_groupAll);
    connect(worker, SIGNAL(finished()), this, SLOT(handleWorkerFinished()));
    connect(worker, SIGNAL(finished()), worker, SLOT(deleteLater()));
    worker->start();
}
void ModelCellProfile::handleWorkerFinished() {
    beginResetModel(); endResetModel();
    emit loadingFinished();
}
void ParseWorker::run() {
    QFile file(m_file);
    if (!file.open(QIODevice::ReadOnly)) return;
    QDataStream in(&file); in.setByteOrder(QDataStream::LittleEndian);
    unsigned int vlen, pid; in >> vlen; file.read(vlen); in >> pid;
    while (!file.atEnd()) {
        unsigned int ref, nlen; in >> ref >> nlen;
        if (in.status() != QDataStream::Ok) break;
        QString name = QString::fromUtf8(file.read(nlen));
        double l, b, r, t; in >> l >> b >> r >> t;
        QRectF rect(l, -t, r - l, t - b);
        unsigned long long d64; unsigned char d8; in >> d64 >> d64 >> d64 >> d8;
        m_allGrp->addChild(new CellEntry(name, rect, m_allGrp));
        if (m_topGrp->children().isEmpty() || name.toUpper().contains("TOP")) {
            m_topGrp->addChild(new CellEntry(name, rect, m_topGrp));
        }
    }
    file.close(); emit finished();
}
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