// #include "tensorflow/core/util/event.pb.h"
// #include <iostream>
// #include <fstream>
// #include <string>
// #include <stdio.h>

// using namespace std;
// using namespace tensorflow;

// void ListEntry(const tensorflow::Event &event)
// {
//   cout << "wall time: " << event.wall_time() << endl;
//   cout << "step: " << event.step() << endl;
//   if (event.has_summary())
//   {
//     cout << "summary" << endl;
//     tensorflow::Summary summary = event.summary();
//     for (int i = 0; i < summary.value_size(); ++i)
//     {
//       tensorflow::Summary::Value value = summary.value(i);
//     }
//     // for (const auto &value : summary.value())
//     // {
//     //   cout << "node name: " << value.node_name() << endl;
//     //   cout << "tag: " << value.tag() << ", Value: " << value.simple_value() << endl;
//     // }
//   }
// }

// int main(int argc, char **argv)
// {

//   std::ifstream file;
//   std::uint64_t current_pos = 0;
//   string path = "/home/justin126/workspace/ShareTest/prototest/logs/train/events.out.tfevents.1737293120.justin126-VirtualBox.24119.0.v2";

//   // Event event;
//   // fstream input(path.c_str(), ios::in | ios::binary);
//   // // cout << sizeof(uint64_t) << endl; // 8bytes
//   // // cout << sizeof(uint32_t) << endl; // 4bytes

//   // uint64_t length = 0;
//   // uint64_t max_len = 1000;
//   // char* data_buffer = new char[max_len];
//   // // Format of a single record:
//   // //  uint64    length
//   // //  uint32    masked crc of length
//   // //  byte      data[length]
//   // //  uint32    masked crc of data

//   // while (input.read(reinterpret_cast<char*>(&length), 8)) {
//   //   if (length > max_len) {
//   //     delete[] data_buffer;
//   //     max_len = 2*length;
//   //     data_buffer = new char[max_len];
//   //   }
//   //   // cout << "packet len: " << length << endl;
//   //   input.read(data_buffer, length+8);

//   //   tensorflow::Event event;
//   //   if (!event.ParseFromArray(reinterpret_cast<char*>(data_buffer+4), length)) {
//   //     std::cerr << "Failed to parse event." << std::endl;
//   //     return -1;
//   //   } else {
//   //     cout << "Parsed event successfully." << endl;
//   //     for (const auto& value : event.summary().value()) {
//   //       cout << "Tag: " << value.tag() << endl;
//   //     }

//   //   }
//   //   ListEntry(event);
//   // }
//   // input.close();
//   std::uint64_t length;
//   std::uint32_t crc;

//   if (!file.is_open())
//   {
//     file.open(path, std::ios::binary);
//     file.seekg(current_pos, std::ios::beg);
//   }

//   current_pos = file.tellg();

//   if (file.peek() == EOF)
//   {
//     file.close();
//   }

//   while (file.read(reinterpret_cast<char *>(&length), sizeof(std::uint64_t)))
//   {
//     if (file.eof())
//     {
//       file.clear();
//     }

//     file.read(reinterpret_cast<char *>(&crc), sizeof(std::uint32_t));

//     std::vector<char> buffer(length);
//     file.read(&buffer[0], length);

//     tensorflow::Event event;
//     if (event.ParseFromString(std::string(buffer.begin(), buffer.end())))
//     {
//       ListEntry(event);
//     }
//     // if (event.ParseFromArray(static_cast<void*>(buffer.data()), length)) {
//     //   ListEntry(event);
//     // }

//     file.read(reinterpret_cast<char *>(&crc), sizeof(std::uint32_t));
//   }

//   return 0;
// }

#include <iostream>
#include <fstream>
#include <string>
#include <tensorflow/core/framework/summary.pb.h>
#include <tensorflow/core/util/event.pb.h>
#include <tensorflow/core/lib/io/record_reader.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/framework/tensor.h>
#include "tensorflow/core/framework/types.h"

using namespace std;
using namespace tensorflow;

int main()
{
  string path="/home/justin126/workspace/ShareTest/prototest/logs/train/events.out.tfevents.1737293120.justin126-VirtualBox.24119.0.v2";
  tensorflow::Env *env = tensorflow::Env::Default();
  std::unique_ptr<tensorflow::RandomAccessFile> file;
  tensorflow::Status status = env->NewRandomAccessFile(path, &file);

  io::SequentialRecordReader record_reader(file.get());
  tstring record;

  while (record_reader.ReadRecord(&record).ok())
  {
    tensorflow::Event event;
    if (!event.ParseFromString(record))
    {
      continue;
    }

    if (event.has_summary()) {
      const tensorflow::Summary summary = event.summary();
      for (const auto& value : summary.value()) {
        cout << "Tag: " << value.tag() << ", Value: " << value.simple_value() << endl;
        // print tensor

        if (value.has_tensor()) {
          cout << "tensror exists: " << endl;
          const auto& tensor = value.tensor();
          cout << "Tensor DType: " << tensor.dtype() << endl;
          Tensor t;
          if (t.FromProto(tensor)) {
            cout << "shape: " << t.shape().DebugString() << endl;

            if (t.dtype() == DT_FLOAT) {
              auto float_tensor = t.flat<float>();
              for (int i = 0; i < float_tensor.size(); ++i) {
                cout << "float value: " << float_tensor(i) << endl;
              }
            } else if (t.dtype() == DT_STRING) {
              auto string_tensor = t.flat<tstring>();
              cout << "Tensor contains strings:" << endl;
              for (int i = 0; i < string_tensor.size(); ++i) {
                cout << "  String[" << i << "]: " << string_tensor(i) << endl;
              }
            }
          }
        }
        
      }
    }
    
  }
}