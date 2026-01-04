#include "ModelCellProfile.h"
#include <fstream>
#include <vector>
#include <stdint.h>

void ParseWorker::run() {
    std::ifstream ifs(m_file.toLocal8Bit().constData(), std::ios::binary);
    if (!ifs.is_open()) return;

    // 1. Header: version, pid, dbu, table_off
    uint32_t vlen; ifs.read((char*)&vlen, 4);
    ifs.seekg(vlen, std::ios::cur); 
    uint32_t pid, dbu; uint64_t t_off;
    ifs.read((char*)&pid, 4); ifs.read((char*)&dbu, 4); ifs.read((char*)&t_off, 8);

    // 2. Cell Records (Type 1)
    while (true) {
        unsigned char type; 
        if (!ifs.read((char*)&type, 1) || type == 0) break; // 3. End Tag 0

        uint32_t cidx, nlen;
        ifs.read((char*)&cidx, 4); ifs.read((char*)&nlen, 4);
        std::string name(nlen, '\0'); ifs.read(&name[0], nlen);
        
        double l, b, r, t;
        ifs.read((char*)&l, 8); ifs.read((char*)&b, 8);
        ifs.read((char*)&r, 8); ifs.read((char*)&t, 8);
        
        // Skip 29 bytes: pos(8)+len(8)+fig(8)+used(4)+flag(1)
        ifs.seekg(29, std::ios::cur);

        m_groupAll->addChild(new CellEntry(QString::fromStdString(name), QRectF(l, -t, r-l, t-b), m_groupAll));
    }

    // 4. Top Cells
    uint32_t top_n; ifs.read((char*)&top_n, 4);
    for(uint32_t i=0; i<top_n; ++i) {
        uint32_t tnlen; ifs.read((char*)&tnlen, 4);
        std::string tname(tnlen, '\0'); ifs.read(&tname[0], tnlen);
        QString qtname = QString::fromStdString(tname);
        for(int j=0; j<m_groupAll->children().size(); ++j) {
            if(m_groupAll->children()[j]->name() == qtname) {
                m_groupTop->addChild(new CellEntry(qtname, m_groupAll->children()[j]->rect(), m_groupTop));
                break;
            }
        }
    }
    ifs.close();
    emit finished();
}

ModelCellProfile::ModelCellProfile(QObject *parent) : QAbstractItemModel(parent) {
    m_rootItem = new CellEntry("Root", QRectF());
    m_groupTop = new CellEntry("Top Cells", QRectF(), m_rootItem);
    m_groupAll = new CellEntry("All Cells", QRectF(), m_rootItem);
    m_rootItem->addChild(m_groupTop);
    m_rootItem->addChild(m_groupAll);
}
ModelCellProfile::~ModelCellProfile() { delete m_rootItem; }
void ModelCellProfile::startLoading(const QString &f) {
    qDeleteAll(m_groupTop->children()); m_groupTop->children().clear();
    qDeleteAll(m_groupAll->children()); m_groupAll->children().clear();
    ParseWorker *w = new ParseWorker(f, m_groupTop, m_groupAll);
    connect(w, SIGNAL(finished()), this, SLOT(onWorkerFinished()));
    connect(w, SIGNAL(finished()), w, SLOT(deleteLater()));
    w->start();
}
void ModelCellProfile::onWorkerFinished() { beginResetModel(); endResetModel(); emit loadingFinished(); }
QModelIndex ModelCellProfile::index(int r, int c, const QModelIndex &p) const {
    CellEntry *pe = itemFromIndex(p); return pe->children().value(r) ? createIndex(r, c, pe->children()[r]) : QModelIndex();
}
QModelIndex ModelCellProfile::parent(const QModelIndex &ch) const {
    if (!ch.isValid()) return QModelIndex();
    CellEntry *ce = itemFromIndex(ch); CellEntry *pe = ce->parent();
    return (pe == m_rootItem) ? QModelIndex() : createIndex(pe->row(), 0, pe);
}
int ModelCellProfile::rowCount(const QModelIndex &p) const { return itemFromIndex(p)->children().count(); }
QVariant ModelCellProfile::data(const QModelIndex &i, int r) const { return (i.isValid() && r == Qt::DisplayRole) ? itemFromIndex(i)->name() : QVariant(); }
CellEntry* ModelCellProfile::itemFromIndex(const QModelIndex &i) const { return i.isValid() ? static_cast<CellEntry*>(i.internalPointer()) : m_rootItem; }