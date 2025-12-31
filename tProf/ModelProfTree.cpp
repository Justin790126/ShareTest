#include "ModelProfTree.h"

#include <fstream>
#include <sstream>
#include <cctype>

// ====================== ctor/dtor ======================
ModelProfTree::ModelProfTree(QObject* parent)
    : QThread(parent)
    , m_filePath("/mnt/data/apiClnt.txt")
    , m_root(std::make_shared<ProfNodeStd>("ROOT"))
    , m_stopRequested(false)
{
}

ModelProfTree::~ModelProfTree()
{
    requestStop();
    wait();
}

// ====================== std helpers ======================
static bool isUnitToken(const std::string& t)
{
    return (t == "s" || t == "ms" || t == "us" || t == "ns");
}

static bool looksNumber(const std::string& t)
{
    // 允許 "--"
    if (t == "--") return true;
    try {
        size_t idx = 0;
        (void)std::stod(t, &idx);
        return idx > 0;
    } catch (...) {
        return false;
    }
}


std::string ModelProfTree::trim(const std::string& s)
{
    size_t b = 0, e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

bool ModelProfTree::startsWith(const std::string& s, const char* prefix)
{
    return s.rfind(prefix, 0) == 0;
}

std::vector<std::string> ModelProfTree::splitWS(const std::string& s)
{
    std::vector<std::string> out;
    std::istringstream iss(s);
    std::string tok;
    while (iss >> tok) out.push_back(tok);
    return out;
}

bool ModelProfTree::shouldSkipLine(const std::string& raw)
{
    std::string t = trim(raw);
    if (t.empty()) return true;

    // 非資料列常見行
    if (t.rfind("====", 0) == 0) return true;
    if (t.rfind("Program start:", 0) == 0) return true;
    if (t.rfind("Program end:", 0) == 0) return true;
    if (t.rfind("TOTAL uptime:", 0) == 0) return true;
    if (t.rfind("Client uptime:", 0) == 0) return true;
    if (t.rfind("Server uptime:", 0) == 0) return true;
    if (t.rfind("Peak memory used", 0) == 0) return true;

    // 表頭
    if (t.find("TOTAL TIME") != std::string::npos && t.find("CPU TIME") != std::string::npos) return true;
    if (t.find("NUM OF CALLS") != std::string::npos && t.find("PERCENTAGE") != std::string::npos) return true;

    return false;
}

int ModelProfTree::calcDepthAndStripPrefix(std::string& line)
{
    if (line.empty() || line[0] != '|')
        return 0;

    // 形式: '|' + '-'*N + rest
    size_t i = 1;
    int dashCount = 0;
    while (i < line.size() && line[i] == '-') {
        ++dashCount;
        ++i;
    }
    int depth = dashCount / 2; // "|--"=1, "|----"=2, ...
    line = trim(line.substr(i));
    return depth;
}

bool ModelProfTree::parseDataLine(const std::string& lineNoPrefix,
                                  std::string& name,
                                  std::string& total,
                                  std::string& cpu,
                                  std::string& calls,
                                  std::string& percent,
                                  std::string& child,
                                  std::string& mem)
{
    std::string line = trim(lineNoPrefix);
    if (line.empty()) return false;
    if (shouldSkipLine(line)) return false;

    std::vector<std::string> tk = splitWS(line);
    if (tk.size() < 6) return false;

    // 這份 log 可能的尾端欄位：
    // A) calls percent child mem   (4 tokens)
    // B) calls percent child       (3 tokens) -> mem 設 "--"
    //
    // time 欄位可能是：
    // total: <num> <unit>
    // cpu  : <num> <unit>
    //
    // 所以整體可能長這樣：
    // name ...  totalNum unit cpuNum unit calls percent child [mem]
    //
    // 我們用兩次嘗試：tailCount=4 先試（含 mem），不行再試 tailCount=3（無 mem）
    for (int tailCount = 4; tailCount >= 3; --tailCount) {
        if ((int)tk.size() < tailCount + 3) continue; // 至少要留給 name/total/cpu

        int n = (int)tk.size();
        int callsIdx = n - tailCount; // calls 在尾端區塊的第一個

        const std::string& callsCand   = tk[callsIdx];
        const std::string& percentCand = tk[callsIdx + 1];
        const std::string& childCand   = tk[callsIdx + 2];
        const std::string  memCand     = (tailCount == 4) ? tk[callsIdx + 3] : std::string("--");

        // 檢查尾端欄位像不像數字（允許 "--"）
        if (!looksNumber(callsCand) || !looksNumber(percentCand) || !looksNumber(childCand) || !looksNumber(memCand))
            continue;

        // 現在要解析 calls 前面的 total/cpu
        // callsIdx 前一個 token index：
        int end = callsIdx - 1;
        if (end < 1) continue;

        std::string totalCand, cpuCand, nameCand;

        // Case 1: total/cpu 帶 unit（各 2 tokens）：
        // ... totalNum unit cpuNum unit calls ...
        if (end >= 4 &&
            isUnitToken(tk[end]) && looksNumber(tk[end - 1]) &&
            isUnitToken(tk[end - 2]) && looksNumber(tk[end - 3]))
        {
            cpuCand   = tk[end - 1] + " " + tk[end];
            totalCand = tk[end - 3] + " " + tk[end - 2];

            // name = tk[0 .. end-4]
            std::ostringstream oss;
            for (int i = 0; i <= end - 4; ++i) {
                if (i) oss << ' ';
                oss << tk[i];
            }
            nameCand = trim(oss.str());
        }
        // Case 2: total/cpu 不帶 unit（各 1 token）：
        // ... total cpu calls ...
        else if (end >= 2 && looksNumber(tk[end]) && looksNumber(tk[end - 1])) {
            cpuCand   = tk[end];
            totalCand = tk[end - 1];

            std::ostringstream oss;
            for (int i = 0; i <= end - 2; ++i) {
                if (i) oss << ' ';
                oss << tk[i];
            }
            nameCand = trim(oss.str());
        }
        else {
            continue;
        }

        if (nameCand.empty()) continue;

        // commit
        name    = nameCand;
        total   = totalCand;
        cpu     = cpuCand;
        calls   = callsCand;
        percent = percentCand;
        child   = childCand;
        mem     = memCand;
        return true;
    }

    return false;
}


std::string ModelProfTree::extractBetweenProfileEquals(const std::string& line)
{
    // "======================METIS WORKER PROFILE======================"
    size_t firstNonEq = line.find_first_not_of('=');
    size_t lastNonEq  = line.find_last_not_of('=');
    if (firstNonEq == std::string::npos || lastNonEq == std::string::npos || lastNonEq < firstNonEq)
        return "";
    return trim(line.substr(firstNonEq, lastNonEq - firstNonEq + 1));
}

void ModelProfTree::parseUptimeLine(const std::string& line, std::string& secondsOut, std::string& percentOut)
{
    secondsOut.clear();
    percentOut.clear();

    size_t colon = line.find(':');
    if (colon == std::string::npos) return;

    std::string rhs = trim(line.substr(colon + 1));

    size_t lp = rhs.find('(');
    size_t rp = rhs.find(')');
    if (lp != std::string::npos && rp != std::string::npos && rp > lp) {
        percentOut = trim(rhs.substr(lp + 1, rp - lp - 1)); // "3.284348%"
        secondsOut = trim(rhs.substr(0, lp));              // "2.495362 s"
    } else {
        secondsOut = rhs; // "75.977385 s"
    }
}

// ====================== run() ======================
void ModelProfTree::run()
{
    m_stopRequested = false;
    emit parseStarted(m_filePath);

    const std::string path = m_filePath.toLocal8Bit().constData();
    std::fstream fs(path.c_str(), std::ios::in);
    if (!fs.is_open()) {
        emit parseFailed(QString("Open file failed: %1").arg(m_filePath));
        return;
    }

    // local tree（建好後一次 assign 到 m_root）
    ProfNodeStd::Ptr rootLocal = std::make_shared<ProfNodeStd>("ROOT");

    // header state machine
    enum HeaderState { HS_NONE=0, HS_EXPECT_TOTAL, HS_EXPECT_CLIENT, HS_EXPECT_SERVER };
    HeaderState hs = HS_NONE;

    // stack[depth] = 該深度最新節點；depth 0 對 ROOT
    std::vector<ProfNodeStd::Ptr> stack;
    stack.reserve(32);
    stack.push_back(rootLocal);

    std::string raw;
    while (std::getline(fs, raw)) {
        if (m_stopRequested) {
            emit parseFailed("Parse aborted by requestStop().");
            return;
        }

        std::string line = trim(raw);

        // --------- header: PROFILE + uptime ---------
        if (startsWith(line, "======================") && line.find("PROFILE") != std::string::npos) {
            std::string title = extractBetweenProfileEquals(line);
            rootLocal->meta().setProfileTitle(title);
            if (!title.empty()) rootLocal->setName(title);
            hs = HS_EXPECT_TOTAL;
            continue;
        }

        if (hs != HS_NONE) {
            if (startsWith(line, "Total uptime:")) {
                std::string sec, dummy;
                parseUptimeLine(line, sec, dummy);
                rootLocal->meta().setTotalUptime(sec);
                hs = HS_EXPECT_CLIENT;
                continue;
            }
            if (startsWith(line, "Client uptime:")) {
                std::string sec, pct;
                parseUptimeLine(line, sec, pct);
                rootLocal->meta().setClientUptime(sec);
                rootLocal->meta().setClientPercent(pct);
                hs = HS_EXPECT_SERVER;
                continue;
            }
            if (startsWith(line, "Server uptime:")) {
                std::string sec, pct;
                parseUptimeLine(line, sec, pct);
                rootLocal->meta().setServerUptime(sec);
                rootLocal->meta().setServerPercent(pct);
                hs = HS_NONE;
                continue;
            }
            // unexpected format -> give up this header sequence
            hs = HS_NONE;
            // 不 continue：讓它往下當普通資料列處理
        }

        // --------- data line to tree ---------
        int depth = calcDepthAndStripPrefix(line); // line is now without prefix

        std::string name, total, cpu, calls, percent, child, mem;
        if (!parseDataLine(line, name, total, cpu, calls, percent, child, mem))
            continue;

        // depth 安全修正
        if (depth < 0) depth = 0;
        if (depth > (int)stack.size()) depth = (int)stack.size();

        int parentDepth = depth;
        if (parentDepth >= (int)stack.size()) parentDepth = (int)stack.size() - 1;

        // 回退 stack（純粹維護結構，這版不做事後 sort，因為我們用插入排序）
        while ((int)stack.size() > parentDepth + 1)
            stack.pop_back();

        ProfNodeStd::Ptr parent = stack[parentDepth];

        ProfNodeStd::Ptr cur = std::make_shared<ProfNodeStd>();
        cur->setName(name);
        cur->setColumns(total, cpu, calls, percent, child, mem);

        // ✅ 組樹當下依 totalTime DESC 插入，children 永遠保持排序
        parent->addChildSortedByTotalTime(cur);

        // 更新 stack 下一層
        if ((int)stack.size() == parentDepth + 1)
            stack.push_back(cur);
        else
            stack[parentDepth + 1] = cur;
    }

    fs.close();

    // commit
    m_root = rootLocal;
    emit parseFinished();
}
