#include "ModelTimeSequenceParser.h"

ModelTimeSequenceParser::ModelTimeSequenceParser(QObject *parent)
    : QThread(parent) {
  // Constructor implementation
}

ModelTimeSequenceParser::~ModelTimeSequenceParser() { ClearSequencePairs(); }

double ModelTimeSequenceParser::string_to_timestamp(const std::string &a) {
  std::tm tm = {};
  double milliseconds = 0.0;

  // Parse the date and time
  std::istringstream ss(a);
  ss >> std::get_time(&tm, "%Y-%m-%d-%H:%M:%S");
  if (ss.fail())
    return -1;

  // Read the milliseconds part
  char dot;
  ss >> dot >> milliseconds; // dot should be '.'
  if (dot != '.')
    milliseconds = 0.0;

  // Get time_t (seconds since epoch)
  std::time_t t = timegm(&tm); // use timegm for UTC, or mktime for local time

  // Combine seconds and milliseconds
  return static_cast<double>(t) + milliseconds / 1000.0;
}

string ModelTimeSequenceParser::timestamp_to_string(double timestamp) {
  std::time_t t = static_cast<std::time_t>(timestamp);
  double milliseconds = (timestamp - t) * 1000.0;

  std::tm tm =
      *std::gmtime(&t); // use gmtime for UTC, or localtime for local time

  char buffer[30];
  std::strftime(buffer, sizeof(buffer), "%Y-%m-%d-%H:%M:%S", &tm);
  return std::string(buffer) + "." +
         std::to_string(static_cast<int>(milliseconds));
}

void ModelTimeSequenceParser::SplitByDel(const string &in, vector<string> &out,
                                         const string &del) {
  size_t start = 0;
  size_t end = in.find(del);
  while (end != string::npos) {
    out.push_back(in.substr(start, end - start));
    start = end + del.length();
    end = in.find(del, start);
  }
  out.push_back(in.substr(start, end));
}

void ModelTimeSequenceParser::ParseHeader() {
  // This function should parse the header of the file
  // For now, we will just print a message
  cout << "Parsing header from file: " << m_sFileName << endl;
  // You can add actual parsing logic here
  // For example, read the first line and store it in m_vsHeader
  if (fp && fp->is_open()) {
    string headerLine;
    if (getline(*fp, headerLine)) {
      // split headerLine by ','
      SplitByDel(headerLine, m_vsHeader, ",");

    } else {
      cerr << "Failed to read header from file: " << m_sFileName << endl;
    }
  } else {
    cerr << "File not opened: " << m_sFileName << endl;
  }
}

void ModelTimeSequenceParser::ClearSequencePairs() {
  for (size_t i = 0; i < m_vTimeSequencePairs.size(); ++i) {
    TimeSequencePair *recvPair = m_vTimeSequencePairs[i].first;
    TimeSequencePair *sendPair = m_vTimeSequencePairs[i].second;

    if (recvPair) {
      delete recvPair; // Delete the receive pair
    }
    if (sendPair) {
      delete sendPair; // Delete the send pair
    }
  }
  m_vTimeSequencePairs.clear();
}

int ModelTimeSequenceParser::Str2ActType(const string &in) {
  // Convert string to action type
  if (in == "SEND")
    return (int)MTS_ACTION_TYPE_SEND;
  if (in == "RECV")
    return (int)MTS_ACTION_TYPE_RECV;
  return (int)MTS_ACTION_TYPE_UNKNOWN; // Default case
}

string ModelTimeSequenceParser::ActType2Str(int actType) {
  // Convert action type to string
  switch (actType) {
  case MTS_ACTION_TYPE_SEND:
    return "SEND";
  case MTS_ACTION_TYPE_RECV:
    return "RECV";
  default:
    return "UNKNOWN"; // Default case
  }
}

int ModelTimeSequenceParser::Str2ActId(const string &in) {
  // use in.find to determin enum
  if (in.find("UtilAct::ENABLE_PROFILE") != string::npos) {
    return (int)MTS_ACTION_ID_ENABLE_PROFILE;
  } else if (in.find("SimAct::LOAD_MODEL") != string::npos) {
    return (int)MTS_ACTION_ID_LOAD_MODEL;
  } else if (in.find("SimAct::MODEL_SETTING") != string::npos) {
    return (int)MTS_ACTION_ID_MODEL_SETTING;
  } else if (in.find("SimAct::DOMAIN_INFO") != string::npos) {
    return (int)MTS_ACTION_ID_DOMAIN_INFO;
  } else if (in.find("SimAct::PRIV_ACT") != string::npos) {
    return (int)MTS_ACTION_ID_PRIV_ACT;
  } else if (in.find("SimAct::SET_MASK") != string::npos) {
    return (int)MTS_ACTION_ID_SET_MASK;
  } else if (in.find("SimAct::COMP_RESIST_IMAGE") != string::npos) {
    return (int)MTS_ACTION_ID_COMP_RESIST_IMAGE;
  } else if (in.find("SimAct::COMP_OPTICAL_IMAGE") != string::npos) {
    return (int)MTS_ACTION_ID_COMP_OPTICAL_IMAGE;
  } else if (in.find("SimAct::SET_NULL_MASK") != string::npos) {
    return (int)MTS_ACTION_ID_SET_NULL_MASK;
  } else if (in.find("SimAct::UNLOAD_MODEL") != string::npos) {
    return (int)MTS_ACTION_ID_UNLOAD_MODEL;
  } else if (in.find("[Unknown SimAct]") != string::npos) {
    return (int)MTS_ACTION_ID_UNKNOWN_SIM_ACT;
  }
}

string ModelTimeSequenceParser::ActId2Str(int actId) {
  // Convert action ID to string
  switch (actId) {
  case MTS_ACTION_ID_ENABLE_PROFILE:
    return "UtilAct::ENABLE_PROFILE";
  case MTS_ACTION_ID_LOAD_MODEL:
    return "SimAct::LOAD_MODEL";
  case MTS_ACTION_ID_MODEL_SETTING:
    return "SimAct::MODEL_SETTING";
  case MTS_ACTION_ID_DOMAIN_INFO:
    return "SimAct::DOMAIN_INFO";
  case MTS_ACTION_ID_PRIV_ACT:
    return "SimAct::PRIV_ACT";
  case MTS_ACTION_ID_SET_MASK:
    return "SimAct::SET_MASK";
  case MTS_ACTION_ID_COMP_RESIST_IMAGE:
    return "SimAct::COMP_RESIST_IMAGE";
  case MTS_ACTION_ID_COMP_OPTICAL_IMAGE:
    return "SimAct::COMP_OPTICAL_IMAGE";
  case MTS_ACTION_ID_SET_NULL_MASK:
    return "SimAct::SET_NULL_MASK";
  case MTS_ACTION_ID_UNLOAD_MODEL:
    return "SimAct::UNLOAD_MODEL";
  case MTS_ACTION_ID_UNKNOWN_SIM_ACT:
    return "[Unknown SimAct]";
  default:
    return "[Unknown Action ID]"; // Default case
  }
}

string ModelTimeSequenceParser::ActId2JetHexColor(int actId) {
  // use MTS_ACTION_ID_COUNT to create jet color map, and look up map
  vector<string> jetColorHex;
  jetColorHex.reserve(MTS_ACTION_ID_COUNT);
  // create jet color map by MTS_ACTION_ID_COUNT through equation
  for (int i = 0; i < MTS_ACTION_ID_COUNT; ++i) {
    double ratio = static_cast<double>(i) / (MTS_ACTION_ID_COUNT - 1);
    int r = static_cast<int>(255 * (1 - ratio));
    int g = static_cast<int>(255 * ratio);
    int b = 0; // Blue is always 0 in this case
    char hexColor[8];
    snprintf(hexColor, sizeof(hexColor), "#%02X%02X%02X", r, g, b);
    jetColorHex.push_back(hexColor);
  }
  jetColorHex.shrink_to_fit(); // Shrink to fit the reserved space

  // return the color by actId
  if (actId >= 0 && actId < MTS_ACTION_ID_COUNT) {
    return jetColorHex[actId];
  } else {
    return "#000000"; // Default color for unknown action ID
  }
}

void ModelTimeSequenceParser::ParseSequencePairs() {
  // iterate m_vvContent
  vector<pair<int, size_t>> recvActIdAndRowIdx;
  vector<pair<int, size_t>> sendActIdAndRowIdx;

  recvActIdAndRowIdx.reserve(1000); // Reserve space for 1000 pairs
  sendActIdAndRowIdx.reserve(1000); // Reserve space for 1000 pairs
  for (size_t i = 0; i < m_vvContent.size(); ++i) {
    const auto &row = m_vvContent[i];
    if (row.size() < 3)
      continue; // Ensure there are enough columns

    double ts = string_to_timestamp(row[0]);
    // printf("Row %zu: Timestamp = %f\n", i, ts);
    int actType = Str2ActType(row[1]);
    int actId = Str2ActId(row[2]);

    if (actType == MTS_ACTION_TYPE_RECV) {
      recvActIdAndRowIdx.emplace_back(actId, i);
    } else if (actType == MTS_ACTION_TYPE_SEND) {
      sendActIdAndRowIdx.emplace_back(actId, i);
    }
  }
  recvActIdAndRowIdx.shrink_to_fit(); // Shrink to fit the reserved space
  sendActIdAndRowIdx.shrink_to_fit(); // Shrink to fit the reserved space

  m_vdTimeStamps.clear();
  m_vdTimeStamps.reserve(m_vvContent.size());
  for (size_t i = 0; i < m_vvContent.size(); ++i) {
    const auto &row = m_vvContent[i];
    if (row.size() < 3)
      continue; // Ensure there are enough columns

    double ts = string_to_timestamp(row[0]);
    m_vdTimeStamps.push_back(ts);
  }
  m_vdTimeStamps.shrink_to_fit(); // Shrink to fit the reserved space

  // sort m_vdTimeStamps by timestamp from small to large
  std::sort(m_vdTimeStamps.begin(), m_vdTimeStamps.end());

  // calculate m_vdNormalizedTimeStamps by minus each timestamp with the first
  // timestamp
  m_vdNormalizedTimeStamps.clear();
  m_vdNormalizedTimeStamps.reserve(m_vdTimeStamps.size());
  if (!m_vdTimeStamps.empty()) {
    double firstTimestamp = m_vdTimeStamps[0];
    for (const auto &ts : m_vdTimeStamps) {
      m_vdNormalizedTimeStamps.push_back(ts - firstTimestamp);
    }
  }
  m_vdNormalizedTimeStamps.shrink_to_fit(); // Shrink to fit the reserved space

  ClearSequencePairs();

  vector<size_t> pairRecvIdxes;
  vector<size_t> pairSendIdxes;

  size_t iterSize =
      std::min<size_t>(recvActIdAndRowIdx.size(), sendActIdAndRowIdx.size());
  pairRecvIdxes.reserve(iterSize);
  pairSendIdxes.reserve(iterSize);
  for (size_t i = 0; i < iterSize; i++) {
    int recvActId = recvActIdAndRowIdx[i].first;
    size_t recvRowIdx = recvActIdAndRowIdx[i].second;

    // explicit iterator to find the corresponding sendActId
    // if recvActId is MTS_ACTION_ID_LOAD_MODEL, find the corresponding
    // sendActId which is MTS_ACTION_ID_UNLOAD_MODEL

    int pairSendIdx = -1;
    // find index where the same act id in sendActIdAndRowIdx
    if (pairSendIdx == -1) {
      auto it =
          std::find_if(sendActIdAndRowIdx.begin(), sendActIdAndRowIdx.end(),
                       [recvActId](const pair<int, size_t> &p) {
                         return p.first == recvActId;
                       });
      if (it != sendActIdAndRowIdx.end()) {
        pairSendIdx = it - sendActIdAndRowIdx.begin(); // get index
      }
    }

    if (pairSendIdx != -1) {
      size_t sendRowIdx = sendActIdAndRowIdx[pairSendIdx].second;
      pairRecvIdxes.push_back(recvRowIdx);
      pairSendIdxes.push_back(sendRowIdx);
    }
  }
  pairRecvIdxes.shrink_to_fit(); // Shrink to fit the reserved space
  pairSendIdxes.shrink_to_fit(); // Shrink to fit the reserved space

  // remain sendActIdAndRowIdx is not pair
  m_vTimeSequencePairs.clear();
  m_vTimeSequencePairs.reserve(m_vvContent.size());
  for (size_t i = 0; i < pairRecvIdxes.size(); i++) {
    // print in m_vvContent
    const auto &recvRow = m_vvContent[pairRecvIdxes[i]];
    const auto &sendRow = m_vvContent[pairSendIdxes[i]];
    double recvTs = string_to_timestamp(recvRow[0]);
    double sendTs = string_to_timestamp(sendRow[0]);
    int recvActType = Str2ActType(recvRow[1]);
    int recvActId = Str2ActId(recvRow[2]);
    int sendActType = Str2ActType(sendRow[1]);
    int sendActId = Str2ActId(sendRow[2]);
    TimeSequencePair *recvPair =
        new TimeSequencePair(recvTs, recvActType, recvActId);
    TimeSequencePair *sendPair =
        new TimeSequencePair(sendTs, sendActType, sendActId);
    m_vTimeSequencePairs.emplace_back(recvPair, sendPair);
    // cout << recvRow[2] << ", " << sendRow[2] << endl;
    // cout << "Recv Pair: " << *recvPair << ", Send Pair: " << *sendPair <<
    // endl;
  }

  // find those are not paired by using pairRecvIdxes and m_vvContent
  for (size_t i = 0; i < m_vvContent.size(); i++) {
    // check i in pairRecvIdxes ?
    if (std::find(pairRecvIdxes.begin(), pairRecvIdxes.end(), i) ==
            pairRecvIdxes.end() &&
        std::find(pairSendIdxes.begin(), pairSendIdxes.end(), i) ==
            pairSendIdxes.end()) {
      // this row is not paired
      const auto &row = m_vvContent[i];
      if (row.size() < 3)
        continue; // Ensure there are enough columns
      double ts = string_to_timestamp(row[0]);
      int actType = Str2ActType(row[1]);
      int actId = Str2ActId(row[2]);
      TimeSequencePair *pair = new TimeSequencePair(ts, actType, actId);
      m_vTimeSequencePairs.emplace_back(pair, nullptr); // Only recv pair
    }
  }
  m_vTimeSequencePairs.shrink_to_fit(); // Shrink to fit the reserved space

  // sort m_vTimeSequencePairs by timestamp from small to large
  std::sort(m_vTimeSequencePairs.begin(), m_vTimeSequencePairs.end(),
            [](const pair<TimeSequencePair *, TimeSequencePair *> &a,
               const pair<TimeSequencePair *, TimeSequencePair *> &b) {
              return a.first->GetTimeStamp() < b.first->GetTimeStamp();
            });

  m_vTimeSequencePairsByActId.clear();
  m_vTimeSequencePairsByActId.reserve(1000);
  for (size_t i = 0; i < m_vTimeSequencePairs.size(); ++i) {
    TimeSequencePair *recvPair = m_vTimeSequencePairs[i].first;
    TimeSequencePair *sendPair = m_vTimeSequencePairs[i].second;

    // recvPair id
    int recvActId = recvPair ? recvPair->GetActId() : -1;
    string recvActIdStr =
        recvPair ? ActId2Str(recvActId) : "[Unknown Recv Act ID]";
    if ((recvPair && sendPair) || (recvPair && !sendPair)) {
      // check recvActIdStr in m_vTimeSequencePairsByActId first
      auto it = std::find_if(
          m_vTimeSequencePairsByActId.begin(),
          m_vTimeSequencePairsByActId.end(),
          [recvActIdStr](
              const pair<string,
                         vector<pair<TimeSequencePair *, TimeSequencePair *>>>
                  &p) { return p.first == recvActIdStr; });
      if (it != m_vTimeSequencePairsByActId.end()) {
        // found, emplace_back the pair
        it->second.emplace_back(recvPair, sendPair ? sendPair : nullptr);
      } else {
        // not found, create a new pair and emplace_back
        vector<pair<TimeSequencePair *, TimeSequencePair *>> newPairs;
        newPairs.emplace_back(recvPair, sendPair ? sendPair : nullptr);
        m_vTimeSequencePairsByActId.emplace_back(recvActIdStr,
                                                 std::move(newPairs));
      }
    }
  }
  m_vTimeSequencePairsByActId.shrink_to_fit(); // Shrink to fit the reserved space

  // show all string in m_vTimeSequencePairsByActId
  cout << "Parsed " << m_vTimeSequencePairsByActId.size() << endl;
}

void ModelTimeSequenceParser::run() {
  // use fp to parse line by line
  if (!fp || !fp->is_open()) {
    cerr << "File not opened: " << m_sFileName << endl;
    return;
  }
  ParseHeader();

  m_vvContent.clear();       // Clear previous content
  m_vvContent.reserve(1000); // Reserve space for 1000 lines
  string line;
  while (getline(*fp, line)) {
    // split by ',' and emplace_back
    vector<string> row;
    SplitByDel(line, row, ",");
    if (!row.empty()) {
      m_vvContent.emplace_back(std::move(row)); // Move the row to avoid copying
    }
  }
  m_vvContent.shrink_to_fit(); // Shrink to fit the reserved space

  ParseSequencePairs();
}

bool ModelTimeSequenceParser::OpenFile(const QString &fileName) {
  m_sFileName = fileName.toStdString();
  fp = new fstream(m_sFileName, ios::in);
  if (!fp->is_open()) {
    cerr << "Error opening file: " << m_sFileName << endl;
    return false;
  }
  return true;
}

void ModelTimeSequenceParser::Wait() {
  while (isRunning()) {
    msleep(100);                   // Sleep for 100 milliseconds
    QApplication::processEvents(); // Process events to keep the UI responsive
  }
}