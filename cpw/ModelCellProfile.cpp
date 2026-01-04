#include "ModelCellProfile.h"
#include <fstream>  // 標準 C++ 檔案流
#include <vector>

ModelCellProfile::ModelCellProfile(QObject *parent) : QAbstractItemModel(parent) {
    m_rootItem = new CellEntry("Root", QRectF());
    m_groupTop = new CellEntry("Top Cells", QRectF(), m_rootItem);
    m_groupAll = new CellEntry("All Cells", QRectF(), m_rootItem);
    m_rootItem->addChild(m_groupTop);
    m_rootItem->addChild(m_groupAll);
}

ModelCellProfile::~ModelCellProfile() { 
    delete m_rootItem; 
}

void ModelCellProfile::startLoading(const QString &fileName) {
    qDeleteAll(m_groupTop->children()); m_groupTop->children().clear();
    qDeleteAll(m_groupAll->children()); m_groupAll->children().clear();
    
    // 將 QString 轉為 std::string 給標準 C++ 使用
    ParseWorker *worker = new ParseWorker(fileName, m_groupTop, m_groupAll);
    connect(worker, SIGNAL(finished()), this, SLOT(handleWorkerFinished()));
    connect(worker, SIGNAL(finished()), worker, SLOT(deleteLater()));
    worker->start();
}

void ModelCellProfile::handleWorkerFinished() {
    beginResetModel(); 
    endResetModel();
    emit loadingFinished();
}

// --- 使用標準 C++ 解析的核心邏輯 ---
void ParseWorker::run() {
    // 1. 使用 std::ifstream 開啟二進位檔
    std::ifstream ifs(m_file.toLocal8Bit().constData(), std::ios::binary);
    if (!ifs.is_open()) return;

    // 2. 解析 Header
    uint32_t vlen = 0;
    ifs.read(reinterpret_cast<char*>(&vlen), sizeof(uint32_t));
    
    // 跳過版本字串 (例如 "Binary_5.2")
    ifs.seekg(vlen, std::ios::cur); 
    
    uint32_t pid = 0;
    ifs.read(reinterpret_cast<char*>(&pid), sizeof(uint32_t));

    // 3. 循環讀取 Cell 數據
    while (ifs.peek() != EOF) {
        uint32_t ref = 0, nlen = 0;
        ifs.read(reinterpret_cast<char*>(&ref), sizeof(uint32_t));
        ifs.read(reinterpret_cast<char*>(&nlen), sizeof(uint32_t));

        if (ifs.gcount() < sizeof(uint32_t)) break; // 讀取失敗或結束

        // 4. 讀取名字 (標準 C++ 做法)
        std::vector<char> nameBuf(nlen + 1, 0);
        ifs.read(&nameBuf[0], nlen);
        QString name = QString::fromUtf8(&nameBuf[0]);

        // 5. 讀取 BBox (4 個 double = 32 bytes)
        double l, b, r, t;
        ifs.read(reinterpret_cast<char*>(&l), sizeof(double));
        ifs.read(reinterpret_cast<char*>(&b), sizeof(double));
        ifs.read(reinterpret_cast<char*>(&r), sizeof(double));
        ifs.read(reinterpret_cast<char*>(&t), sizeof(double));
        
        QRectF rect(l, -t, r - l, t - b);

        // 6. 跳過後段 25 bytes (8+8+8+1)
        // 使用 seekg 跳轉比讀入記憶體更快
        ifs.seekg(25, std::ios::cur);

        // 7. 分堆邏輯
        m_allGrp->addChild(new CellEntry(name, rect, m_allGrp));
        if (m_topGrp->children().isEmpty() || name.toUpper().contains("TOP")) {
            m_topGrp->addChild(new CellEntry(name, rect, m_topGrp));
        }
    }

    ifs.close();
    emit finished();
}

// --- 以下 Model 實作保持不變 ---
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