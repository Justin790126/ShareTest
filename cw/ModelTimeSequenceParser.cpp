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

void ModelTimeSequenceParser::ClearSequencePairs() {}

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

int ModelTimeSequenceParser::actIdStrInVec(
    const string &actIdStr,
    vector<pair<string, vector<pair<TimeSequencePair *, TimeSequencePair *>>>>
        &vec) {

  int res = -1;
  for (size_t i = 0; i < vec.size(); ++i) {
    if (vec[i].first == actIdStr) {
      res = static_cast<int>(i);
      break; // Found the actIdStr, exit the loop
    }
  }
  return res; // Return the index or -1 if not found
}

void ModelTimeSequenceParser::ParseSequencePairs() {

  /*
    Collect m_vdTimeStamps
  */
  m_vdTimeStamps.clear(); // Clear previous timestamps
  m_vdTimeStamps.reserve(m_vvContent.size()); // Reserve space for timestamps
  for (size_t i = 0; i < m_vvContent.size(); ++i) {
    double ts = string_to_timestamp(m_vvContent[i][0]);
    if (ts >= 0) { // Only add valid timestamps
      m_vdTimeStamps.push_back(ts);
    }
  }
  /*
    Collect actIds from m_vvContent and store them in a vector.
  */
  vector<string> actIdStrVec;
  actIdStrVec.reserve(MTS_ACTION_ID_COUNT);
  
  for (size_t i = 0; i < m_vvContent.size(); ++i) {
    string actIdStr = m_vvContent[i][2];
    // check if actIdStr is in actIdStrVec
    if (std::find(actIdStrVec.begin(), actIdStrVec.end(), actIdStr) ==
        actIdStrVec.end()) {
      // not found, add to actIdStrVec
      actIdStrVec.push_back(actIdStr);
      // cout << actIdStr << endl;
    }
  }
  actIdStrVec.shrink_to_fit(); // Shrink to fit the reserved space
  cout << "Num of api : " << actIdStrVec.size() << endl;
  /*
    Collect TimeSquencePair pointers by actIdStr
  */
  vector<pair<string, vector<TimeSequencePair*>>> timeSequencePairsByActId;
  timeSequencePairsByActId.resize(actIdStrVec.size());
  // reserve
  for (size_t i = 0; i < actIdStrVec.size(); ++i) {
    timeSequencePairsByActId[i].first = actIdStrVec[i];
    timeSequencePairsByActId[i].second.reserve(100); // Reserve space for 100 pairs
  }
  for (size_t i = 0; i < m_vvContent.size(); ++i) {
    string actIdStr = m_vvContent[i][2];
    int idx = actIdStrInVec(actIdStr, timeSequencePairsByActId);
    if (idx != -1) {
      double ts = string_to_timestamp(m_vvContent[i][0]);
      int actType = Str2ActType(m_vvContent[i][1]);
      int actId = Str2ActId(m_vvContent[i][2]);
      TimeSequencePair* pair = new TimeSequencePair(ts, actType, actId);
      pair->SetTimeStampStr(m_vvContent[i][0]);

      // Add the pair to the vector at the found index
      timeSequencePairsByActId[idx].second.push_back(pair);
    }
  }
  /*
    iterate timeSequencePairsByActId and pair consecutive RECV/SEND pairs
  */
  for (size_t i = 0; i < timeSequencePairsByActId.size(); ++i) {
    timeSequencePairsByActId[i].second.shrink_to_fit();
  }

  m_vTimeSequencePairsByActId.clear();
  m_vTimeSequencePairsByActId.resize(actIdStrVec.size());
  for (size_t i = 0; i < actIdStrVec.size(); ++i) {
    m_vTimeSequencePairsByActId[i].first = actIdStrVec[i];
    m_vTimeSequencePairsByActId[i].second.reserve(
        timeSequencePairsByActId[i].second.size()); // Reserve space for pairs
  }
  
  for (size_t i = 0; i < timeSequencePairsByActId.size(); ++i) {
    // iterate through the pairs and pair consecutive RECV/SEND pairs
    vector<TimeSequencePair*>& pairs = timeSequencePairsByActId[i].second;
    for (size_t j = 0; j < pairs.size(); ++j) {
      if (pairs[j]->GetActType() == MTS_ACTION_TYPE_RECV) {
        // find the next send
        for (size_t k = j + 1; k < pairs.size(); ++k) {
          if (pairs[k]->GetActType() == MTS_ACTION_TYPE_SEND) {
            // Found a SEND after RECV, create a pair
            m_vTimeSequencePairsByActId[i].second.emplace_back(
                make_pair(pairs[j], pairs[k]));
            j = k; // Move j to k to skip the SEND we just paired
            break; // Exit the inner loop
          }
        }
        // if next is not SEND, create a pair with nullptr
        if (j < pairs.size() && pairs[j]->GetActType() == MTS_ACTION_TYPE_RECV) {
          m_vTimeSequencePairsByActId[i].second.emplace_back(
              make_pair(pairs[j], nullptr)); // Pair with nullptr for SEND
        }
      }
      // if the pair is SEND, and no received found
      else if (pairs[j]->GetActType() == MTS_ACTION_TYPE_SEND) {
        m_vTimeSequencePairsByActId[i].second.emplace_back(
            make_pair(pairs[j], nullptr)); // Pair with nullptr for RECV
      }
    }
  }
  for (size_t i = 0; i < actIdStrVec.size(); ++i) {
    m_vTimeSequencePairsByActId[i].second.shrink_to_fit();
  }
  /*
    Iterate and show m_vTimeSequencePairsByActId
  */
  // for (const auto& actIdPairs : m_vTimeSequencePairsByActId) {
  //   cout << "Action ID: " << actIdPairs.first << endl;
  //   for (const auto& pair : actIdPairs.second) {
  //     cout << "  RECV: " << pair.first->GetTimeStampStr()
  //          << ", SEND: " << (pair.second ? pair.second->GetTimeStampStr() : "N/A")
  //          << endl;
  //   }
  // }

}

int ModelTimeSequenceParser::actIdStrInVec(const string& actIdStr, 
                        vector<pair<string, vector<TimeSequencePair*>>>& vec)
{
  int res = -1;
  for (size_t i = 0; i < vec.size(); ++i) {
    if (vec[i].first == actIdStr) {
      res = static_cast<int>(i);
      break; // Found the actIdStr, exit the loop
    }
  }
  return res; // Return the index or -1 if not found

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