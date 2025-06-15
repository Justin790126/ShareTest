#ifndef MODEL_TIME_SEQUENCE_PARSER_H
#define MODEL_TIME_SEQUENCE_PARSER_H

#include <QApplication>
#include <QThread>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
using namespace std;

enum MTS_ACTION_TYPE {
  MTS_ACTION_TYPE_UNKNOWN,
  MTS_ACTION_TYPE_SEND,
  MTS_ACTION_TYPE_RECV,
};

enum MTS_ACTION_ID {
  MTS_ACTION_ID_ENABLE_PROFILE,
  MTS_ACTION_ID_LOAD_MODEL,
  MTS_ACTION_ID_MODEL_SETTING,
  MTS_ACTION_ID_DOMAIN_INFO,
  MTS_ACTION_ID_PRIV_ACT,
  MTS_ACTION_ID_SET_MASK,
  MTS_ACTION_ID_COMP_RESIST_IMAGE,
  MTS_ACTION_ID_COMP_OPTICAL_IMAGE,
  MTS_ACTION_ID_SET_NULL_MASK,
  MTS_ACTION_ID_UNLOAD_MODEL,
  MTS_ACTION_ID_UNKNOWN_SIM_ACT,
  MTS_ACTION_ID_COUNT
};

class TimeSequencePair {
public:
  TimeSequencePair(double dTimeStamp, int iActType, int iActId)
      : m_dTimeStamp(dTimeStamp), m_iActType(iActType), m_iActId(iActId) {}

  ~TimeSequencePair() = default;

  double GetTimeStamp() const { return m_dTimeStamp; }
  string GetTimeStampStr() const { return m_sTime; }
  int GetActType() const { return m_iActType; }
  int GetActId() const { return m_iActId; }

  void SetTimeStamp(double dTimeStamp) { m_dTimeStamp = dTimeStamp; }
  void SetTimeStampStr(const string &sTime) { m_sTime = sTime; }
  void SetActType(int iActType) { m_iActType = iActType; }
  void SetActId(int iActId) { m_iActId = iActId; }

  // overload cout
  friend std::ostream &operator<<(std::ostream &os,
                                  const TimeSequencePair &pair) {
    os << std::fixed << std::setprecision(15)
       << "TimeStamp: " << pair.m_dTimeStamp << ", ActType: " << pair.m_iActType
       << ", ActId: " << pair.m_iActId;
    return os;
  }

private:
  double m_dTimeStamp;
  string m_sTime;
  int m_iActType;
  int m_iActId;
};

class ModelTimeSequenceParser : public QThread {
  Q_OBJECT
public:
  ModelTimeSequenceParser(QObject *parent = nullptr);
  ~ModelTimeSequenceParser();

  bool OpenFile(const QString &fileName);
  void Wait();

  void ParseHeader();

  void SplitByDel(const string &in, vector<string> &out,
                  const string &del = ",");

  void ParseSequencePairs();

  int Str2ActId(const string &in);
  string ActId2Str(int iActId);
  string ActId2JetHexColor(int iActId);
  int Str2ActType(const string &in);
  string ActType2Str(int iActType);
  double string_to_timestamp(const std::string &a);
  string timestamp_to_string(double timestamp);

  void ClearSequencePairs();

  // get pointer to m_vdTimeStamps
  vector<double> *GetTimeStamps() { return &m_vdTimeStamps; }
  // get pointer to m_vdNormalizedTimeStamps
  vector<double> *GetNormalizedTimeStamps() {
    return &m_vdNormalizedTimeStamps;
  }
  // get pointer to m_vTimeSequencePairsByActId
  vector<pair<string, vector<pair<TimeSequencePair *, TimeSequencePair *>>>> *
  GetTimeSequencePairsByActId() {
    return &m_vTimeSequencePairsByActId;
  }

protected:
  virtual void run() override;

  fstream *fp;

  string m_sFileName;

  vector<string> m_vsHeader;
  vector<vector<string>> m_vvContent;
  vector<double> m_vdTimeStamps;
  vector<double> m_vdNormalizedTimeStamps;

  vector<pair<string, vector<pair<TimeSequencePair *, TimeSequencePair *>>>>
      m_vTimeSequencePairsByActId;
  int actIdStrInVec(
      const string &actIdStr,
      vector<pair<string, vector<pair<TimeSequencePair *, TimeSequencePair *>>>>
          &vec);
  int actIdStrInVec(const string &actIdStr,
                    vector<pair<string, vector<TimeSequencePair *>>> &vec);
};

#endif /* MODEL_TIME_SEQUENCE_PARSER_H */