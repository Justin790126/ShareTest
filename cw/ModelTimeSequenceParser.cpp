#include "ModelTimeSequenceParser.h"

ModelTimeSequenceParser::ModelTimeSequenceParser(QObject *parent)
    : QThread(parent) {
  // Constructor implementation
}

ModelTimeSequenceParser::~ModelTimeSequenceParser() {
  ClearSequencePairs();
}

double ModelTimeSequenceParser::string_to_timestamp(const std::string& a) {
    std::tm tm = {};
    double milliseconds = 0.0;

    // Parse the date and time
    std::istringstream ss(a);
    ss >> std::get_time(&tm, "%Y-%m-%d-%H:%M:%S");
    if (ss.fail()) return -1;

    // Read the milliseconds part
    char dot;
    ss >> dot >> milliseconds; // dot should be '.'
    if (dot != '.') milliseconds = 0.0;

    // Get time_t (seconds since epoch)
    std::time_t t = timegm(&tm); // use timegm for UTC, or mktime for local time

    // Combine seconds and milliseconds
    return static_cast<double>(t) + milliseconds / 1000.0;
}

string ModelTimeSequenceParser::timestamp_to_string(double timestamp) {
    std::time_t t = static_cast<std::time_t>(timestamp);
    double milliseconds = (timestamp - t) * 1000.0;

    std::tm tm = *std::gmtime(&t); // use gmtime for UTC, or localtime for local time

    char buffer[30];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%d-%H:%M:%S", &tm);
    return std::string(buffer) + "." + std::to_string(static_cast<int>(milliseconds));
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

void ModelTimeSequenceParser::ClearSequencePairs()
{
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
    if (in == "SEND") return (int)MTS_ACTION_TYPE_SEND;
    if (in == "RECV") return (int)MTS_ACTION_TYPE_RECV;
    return (int)MTS_ACTION_TYPE_UNKNOWN; // Default case
}

int ModelTimeSequenceParser::Str2ActId(const string &in) {
    // use in.find to determin enum
    if (in.find("ENABLE_PROFILE") != string::npos) {
        return (int)MTS_ACTION_ID_ENABLE_PROFILE;
    } else if (in.find("LOAD_MODEL") != string::npos) {
        return (int)MTS_ACTION_ID_LOAD_MODEL;
    } else if (in.find("MODEL_SETTING") != string::npos) {
        return (int)MTS_ACTION_ID_MODEL_SETTING;
    } else if (in.find("DOMAIN_INFO") != string::npos) {
        return (int)MTS_ACTION_ID_DOMAIN_INFO;
    } else if (in.find("PRIV_ACT") != string::npos) {
        return (int)MTS_ACTION_ID_PRIV_ACT;
    } else if (in.find("SET_MASK") != string::npos) {
        return (int)MTS_ACTION_ID_SET_MASK;
    } else if (in.find("COMP_RESIST_IMAGE") != string::npos) {
        return (int)MTS_ACTION_ID_COMP_RESIST_IMAGE;
    } else if (in.find("COMP_OPTICAL_IMAGE") != string::npos) {
        return (int)MTS_ACTION_ID_COMP_OPTICAL_IMAGE;
    } else if (in.find("SET_NULL_MASK") != string::npos) {
        return (int)MTS_ACTION_ID_SET_NULL_MASK;
    } else if (in.find("UNLOAD_MODEL") != string::npos) {
        return (int)MTS_ACTION_ID_UNLOAD_MODEL;
    } else if (in.find("Unknown SimAct") != string::npos) {
        return (int)MTS_ACTION_ID_UNKNOWN_SIM_ACT;
    }
}

void ModelTimeSequenceParser::ParseSequencePairs()
{
    // iterate m_vvContent
    vector<pair<int, size_t>> recvActIdAndRowIdx;
    vector<pair<int, size_t>> sendActIdAndRowIdx;

    recvActIdAndRowIdx.reserve(1000); // Reserve space for 1000 pairs
    sendActIdAndRowIdx.reserve(1000); // Reserve space for 1000 pairs
    for (size_t i = 0; i < m_vvContent.size(); ++i) {
        const auto &row = m_vvContent[i];
        if (row.size() < 3) continue; // Ensure there are enough columns

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

    cout << recvActIdAndRowIdx.size() << endl;
    cout << sendActIdAndRowIdx.size() << endl;

    ClearSequencePairs();

    vector<size_t> pairRecvIdx;
    vector<size_t> pairSendIdx;
    size_t iterSize = std::min<size_t>(recvActIdAndRowIdx.size(), sendActIdAndRowIdx.size());
    for (size_t i = 0; i < iterSize; i++) {
        int recvActId = recvActIdAndRowIdx[i].first;
        size_t recvRowIdx = recvActIdAndRowIdx[i].second;

        // explicit iterator to find the corresponding sendActId
        // if recvActId is MTS_ACTION_ID_LOAD_MODEL, find the corresponding
        // sendActId which is MTS_ACTION_ID_UNLOAD_MODEL

        int pairSendIdx = -1;
        if (recvActId == (int)MTS_ACTION_ID_LOAD_MODEL) {
            // find index where sendActId == MTS_ACTION_ID_UNLOAD_MODEL
            auto it = std::find_if(sendActIdAndRowIdx.begin(), sendActIdAndRowIdx.end(),
                        [](const pair<int, size_t> &p) {
                            return p.first == (int)MTS_ACTION_ID_UNLOAD_MODEL;
                        });
            if (it != sendActIdAndRowIdx.end()) {
                pairSendIdx = it - sendActIdAndRowIdx.begin(); // get index
            }

        }

        // find index where the same act id in sendActIdAndRowIdx
        if (pairSendIdx == -1) {
            auto it = std::find_if(sendActIdAndRowIdx.begin(), sendActIdAndRowIdx.end(),
                        [recvActId](const pair<int, size_t> &p) {
                            return p.first == recvActId;
                        });
            if (it != sendActIdAndRowIdx.end()) {
                pairSendIdx = it - sendActIdAndRowIdx.begin(); // get index
            }
        }
            
        // if find, create a pair<TimeSequencePair*, TimeSequencePair*>
        if (pairSendIdx != -1) {
            TimeSequencePair *recvPair = new TimeSequencePair(
                string_to_timestamp(m_vvContent[recvRowIdx][0]),
                Str2ActType(m_vvContent[recvRowIdx][1]),
                recvActId);

            TimeSequencePair *sendPair = new TimeSequencePair(
                string_to_timestamp(m_vvContent[sendActIdAndRowIdx[pairSendIdx].second][0]),
                Str2ActType(m_vvContent[sendActIdAndRowIdx[pairSendIdx].second][1]),
                sendActIdAndRowIdx[pairSendIdx].first);

            m_vTimeSequencePairs.emplace_back(recvPair, sendPair);

            // remove by pairSendIdx in sendActIdAndRowIdx
            sendActIdAndRowIdx.erase(sendActIdAndRowIdx.begin() + pairSendIdx);
        } else {
            // not paired, only receive pair
            TimeSequencePair *recvPair = new TimeSequencePair(
                string_to_timestamp(m_vvContent[recvRowIdx][0]),
                Str2ActType(m_vvContent[recvRowIdx][1]),
                recvActId);
            m_vTimeSequencePairs.emplace_back(recvPair, nullptr); // No send pair
        }
    }

    // remain sendActIdAndRowIdx is not pair
    cout << sendActIdAndRowIdx.size() << endl;

    // use iterSize to check which one is not iterate
    if (recvActIdAndRowIdx.size() > iterSize) {
        // push into m_vTimeSequencePairs
        for (size_t i = iterSize; i < recvActIdAndRowIdx.size(); i++) {
            int recvActId = recvActIdAndRowIdx[i].first;
            size_t recvRowIdx = recvActIdAndRowIdx[i].second;

            TimeSequencePair *recvPair = new TimeSequencePair(
                string_to_timestamp(m_vvContent[recvRowIdx][0]),
                Str2ActType(m_vvContent[recvRowIdx][1]),
                recvActId);

            m_vTimeSequencePairs.emplace_back(recvPair, nullptr); // No send pair
        }
    }
    if (sendActIdAndRowIdx.size() > iterSize) {
        // push into m_vTimeSequencePairs
        for (size_t i = iterSize; i < sendActIdAndRowIdx.size(); i++) {
            int sendActId = sendActIdAndRowIdx[i].first;
            size_t sendRowIdx = sendActIdAndRowIdx[i].second;

            TimeSequencePair *sendPair = new TimeSequencePair(
                string_to_timestamp(m_vvContent[sendRowIdx][0]),
                Str2ActType(m_vvContent[sendRowIdx][1]),
                sendActId);

            m_vTimeSequencePairs.emplace_back(nullptr, sendPair); // No receive pair
        }
    }

    // iterate m_vTimeSequencePairs and print
    for (const auto &pair : m_vTimeSequencePairs) {
        if (pair.first && pair.second) {
            cout << *pair.first << " : " << *pair.second << endl;
        } else if (pair.first) {
            cout << *pair.first << " : No send pair" << endl;
        } else if (pair.second) {
            cout << "No receive pair : " << *pair.second << endl;
        } else {
            cout << "No pairs" << endl;
        }
    }
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