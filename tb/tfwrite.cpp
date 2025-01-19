#include "tensorflow/core/framework/summary.h"
#include "tensorflow/core/util/event_writer.h"

int main() {
  // Create an EventWriter to write to a tfevents file.
  tensorflow::EventWriter writer("/tmp/my_logs");

  // Create a Summary object.
  tensorflow::Summary summary;
  tensorflow::Summary::Value* value = summary.mutable_value()->Add();
  value->set_tag("loss");
  value->set_simple_value(0.5); 

  // Create an Event object.
  tensorflow::Event event;
  event.set_wall_time(std::time(nullptr)); // Set current wall time
  event.set_step(0); // Set the step number
  *event.mutable_summary() = summary; 

  // Write the Event to the tfevents file.
  writer.WriteEvent(event);

  // Close the EventWriter.
  writer.Close();

  return 0;
}