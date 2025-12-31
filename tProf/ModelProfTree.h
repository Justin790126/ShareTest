#pragma once

#include <QString>
#include <QThread>

#include <memory>
#include <ostream>
#include <string>
#include <vector>

// ====================== RootMetaStd ======================
class RootMetaStd {
public:
  const std::string &profileTitle() const { return m_profileTitle; }
  const std::string &totalUptime() const { return m_totalUptime; }
  const std::string &clientUptime() const { return m_clientUptime; }
  const std::string &clientPercent() const { return m_clientPercent; }
  const std::string &serverUptime() const { return m_serverUptime; }
  const std::string &serverPercent() const { return m_serverPercent; }

  void setProfileTitle(const std::string &s) { m_profileTitle = s; }
  void setTotalUptime(const std::string &s) { m_totalUptime = s; }
  void setClientUptime(const std::string &s) { m_clientUptime = s; }
  void setClientPercent(const std::string &s) { m_clientPercent = s; }
  void setServerUptime(const std::string &s) { m_serverUptime = s; }
  void setServerPercent(const std::string &s) { m_serverPercent = s; }

  friend std::ostream &operator<<(std::ostream &os, const RootMetaStd &m) {
    if (!m.m_profileTitle.empty())
      os << "PROFILE: " << m.m_profileTitle << "\n";
    if (!m.m_totalUptime.empty())
      os << "Total uptime : " << m.m_totalUptime << "\n";
    if (!m.m_clientUptime.empty())
      os << "Client uptime: " << m.m_clientUptime << " (" << m.m_clientPercent
         << ")\n";
    if (!m.m_serverUptime.empty())
      os << "Server uptime: " << m.m_serverUptime << " (" << m.m_serverPercent
         << ")\n";
    return os;
  }

private:
  std::string m_profileTitle;
  std::string m_totalUptime;
  std::string m_clientUptime, m_clientPercent;
  std::string m_serverUptime, m_serverPercent;
};

// ====================== ProfNodeStd ======================
class ProfNodeStd {
public:
  typedef std::shared_ptr<ProfNodeStd> Ptr;

  ProfNodeStd() = default;
  explicit ProfNodeStd(const std::string &name) : m_name(name) {}

  const std::string &name() const { return m_name; }
  void setName(const std::string &s) { m_name = s; }

  const std::string &totalTimeMs() const { return m_totalTimeMs; }
  const std::string &cpuTimeMs() const { return m_cpuTimeMs; }
  const std::string &numCalls() const { return m_numCalls; }
  const std::string &percentage() const { return m_percentage; }
  const std::string &childProf() const { return m_childProf; }
  const std::string &memoryMb() const { return m_memoryMb; }

  void setColumns(const std::string &total, const std::string &cpu,
                  const std::string &calls, const std::string &percent,
                  const std::string &child, const std::string &mem) {
    m_totalTimeMs = total;
    m_cpuTimeMs = cpu;
    m_numCalls = calls;
    m_percentage = percent;
    m_childProf = child;
    m_memoryMb = mem;
  }

  RootMetaStd &meta() { return m_meta; }
  const RootMetaStd &meta() const { return m_meta; }

  const std::vector<Ptr> &children() const { return m_children; }

  // ✅ 依 totalTime DESC 插入 child（組樹當下就保持排序）
  void addChildSortedByTotalTime(const Ptr &child) {
    if (!child)
      return;

    // ★ 規則 1：TOTAL summary 不參與排序，永遠放最後
    if (child->name() == "TOTAL") {
      m_children.push_back(child);
      return;
    }

    // ★ 找出目前 children 中 TOTAL 的位置（如果存在）
    // 所有非 TOTAL 的節點，只能插在 TOTAL 之前
    auto totalPos = m_children.end();
    for (auto it = m_children.begin(); it != m_children.end(); ++it) {
      if (*it && (*it)->name() == "TOTAL") {
        totalPos = it;
        break;
      }
    }

    const double childT = toDoubleSafe(child->totalTimeMs());

    // ★ 在 [begin, totalPos) 範圍內做 totalTime DESC 插入
    auto it = m_children.begin();
    for (; it != totalPos; ++it) {
      const double curT = toDoubleSafe((*it)->totalTimeMs());
      if (childT > curT)
        break;
    }

    m_children.insert(it, child);
  }

  // 方便 debug：印整棵 subtree
  friend std::ostream &operator<<(std::ostream &os, const ProfNodeStd &root) {
    if (!root.m_meta.profileTitle().empty())
      os << root.m_meta << "\n";

    for (size_t i = 0; i < root.m_children.size(); ++i)
      dump(os, root.m_children[i], 0);

    return os;
  }

private:
  static double toDoubleSafe(const std::string &s) {
    try {
      size_t idx = 0;
      double v = std::stod(s, &idx);
      (void)idx;
      return v;
    } catch (...) {
      return 0.0;
    }
  }

  static void dump(std::ostream &os, const Ptr &n, int depth) {
    for (int i = 0; i < depth; ++i)
      os << "  ";
    os << n->m_name;
    if (!n->m_totalTimeMs.empty()) {
      os << " | total=" << n->m_totalTimeMs << " cpu=" << n->m_cpuTimeMs
         << " calls=" << n->m_numCalls << " %=" << n->m_percentage
         << " child%=" << n->m_childProf << " mem=" << n->m_memoryMb;
    }
    os << "\n";
    for (size_t i = 0; i < n->m_children.size(); ++i)
      dump(os, n->m_children[i], depth + 1);
  }

private:
  std::string m_name;
  std::string m_totalTimeMs;
  std::string m_cpuTimeMs;
  std::string m_numCalls;
  std::string m_percentage;
  std::string m_childProf;
  std::string m_memoryMb;

  RootMetaStd m_meta;          // root 用
  std::vector<Ptr> m_children; // siblings 永遠保持 totalTime DESC
};

// ====================== ModelProfTree ======================
class ModelProfTree : public QThread {
  Q_OBJECT
public:
  explicit ModelProfTree(QObject *parent = 0);
  virtual ~ModelProfTree();

  void setFilePath(const QString &path) { m_filePath = path; }
  QString filePath() const { return m_filePath; }

  // 建議在 parseFinished() 後再取（你保證一次只有一條 parse thread）
  std::shared_ptr<const ProfNodeStd> rootStd() const { return m_root; }

  void requestStop() { m_stopRequested = true; }

signals:
  void parseStarted(const QString &path);
  void parseFinished();
  void parseFailed(const QString &msg);

protected:
  virtual void run();

private:
  // ===== std C++ parse helpers (NO Qt) =====
  static std::string trim(const std::string &s);
  static bool startsWith(const std::string &s, const char *prefix);
  static std::vector<std::string> splitWS(const std::string &s);

  static bool shouldSkipLine(const std::string &line);

  // 解析 "|----" depth，並 strip prefix（line 會被改寫）
  static int calcDepthAndStripPrefix(std::string &line);

  // 解析資料列：name + 6 欄（允許 "--"）
  static bool parseDataLine(const std::string &lineNoPrefix, std::string &name,
                            std::string &total, std::string &cpu,
                            std::string &calls, std::string &percent,
                            std::string &child, std::string &mem);

  // 解析 "======================...PROFILE======================"
  static std::string extractBetweenProfileEquals(const std::string &line);

  // 解析 uptime 行（Total/Client/Server）
  static void parseUptimeLine(const std::string &line, std::string &secondsOut,
                              std::string &percentOut);

private:
  QString m_filePath;
  ProfNodeStd::Ptr m_root;
  volatile bool m_stopRequested;
};
