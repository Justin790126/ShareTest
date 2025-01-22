#include "tensorflow/core/util/event.pb.h" 
#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>

using namespace std;
using namespace tensorflow;

void ListEntry(const tensorflow::Event& event) {
  cout << "wall time: " << event.wall_time() << endl;
  cout << "step: " << event.step() << endl;
  // has summary ?
  // if (event.has_summary()) {
  //   const auto& summary = event.summary();
  //   cout << "summary size: " << summary.value_size() << endl;
  //   for (const auto& value : summary.value()) {
  //     cout << "Tag: " << value.tag() << ", Value: " << value.simple_value() << endl;
  //   }
  // }
  // print tensor
  
}

int main(int argc, char** argv) {

  string path = "/home/justin126/workspace/ShareTest/prototest/logs/train/events.out.tfevents.1737293120.justin126-VirtualBox.24119.0.v2";

  // Event event;
  fstream input(path.c_str(), ios::in | ios::binary);
  // cout << sizeof(uint64_t) << endl; // 8bytes
  // cout << sizeof(uint32_t) << endl; // 4bytes


  uint64_t length = 0;
  uint64_t max_len = 1000;
  char* data_buffer = new char[max_len];
  // Format of a single record:
  //  uint64    length
  //  uint32    masked crc of length
  //  byte      data[length]
  //  uint32    masked crc of data

  while (input.read(reinterpret_cast<char*>(&length), 8)) {
    if (length > max_len) {
      delete[] data_buffer;
      max_len = 2*length;
      data_buffer = new char[max_len];
    }
    // cout << "packet len: " << length << endl;
    input.read(data_buffer, length+8);

    tensorflow::Event event;
    if (!event.ParseFromArray(reinterpret_cast<char*>(data_buffer+4), length)) {
      std::cerr << "Failed to parse event." << std::endl;
      return -1;
    } else {
      cout << "Parsed event successfully." << endl;
      // for (const auto& value : event.summary().value()) {
      //   cout << "Tag: " << value.tag() << ", Value: " << value.simple_value() << endl;
      // }
      
    }
    ListEntry(event);
  }
  input.close();
 
  return 0;
}