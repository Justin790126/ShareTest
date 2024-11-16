// #include <yaml-cpp/yaml.h>
// #include <iostream>
// #include <stdlib.h>

// using namespace std;

#include <QApplication>
#include <QWidget>
#include "ViewManual.h"

int main(int argc, char *argv[]) {
    QApplication a(argc, argv);
    ViewManual* wid = new ViewManual;
    wid->resize(640, 480);
    wid->show();
    // Webpage drawingWidget;
    // drawingWidget.resize(400, 300);
    // drawingWidget.show();
    return a.exec();
}


// int main() {
//     YAML::Node config = YAML::LoadFile("test1.yaml");

//     // Accessing scalar values
//     std::string name = config["name"].as<std::string>();
//     int age = config["age"].as<int>();
//     std::string city = config["city"].as<std::string>();
//     printf("name = %s \n age = %d \n city = %s \n",
//      name.c_str(), age, city.c_str());
//     // Accessing a sequence (list)
//     const YAML::Node& hobbies = config["hobbies"];
//     printf("hobbies:\n");
//     for (const auto& hobby : hobbies) {
//         std::cout << "-" << hobby.as<std::string>() << std::endl;
//     }
//     printf("person\n");
//     string person_name = config["person"]["name"].as<std::string>();
//     printf("    name: %s \n", person_name.c_str());
//     int person_age = config["person"]["age"].as<int>();
//     printf("    age: %d \n", person_age);
//     YAML::Node address_node = config["person"]["address"];
//     string address_street = address_node["street"].as<std::string>();
//     printf("address\n");
//     printf("    street: %s\n", address_street.c_str());
//     string address_city = address_node["city"].as<std::string>();
//     printf("    city: %s\n", address_city.c_str());
//     string address_state = address_node["state"].as<std::string>();
//     printf("    state: %s\n", address_state.c_str());
//     int address_zip = address_node["zip"].as<int>();
//     printf("    zip: %d\n", address_zip);
//     printf("contact:\n");
//     YAML::Node contact_node = config["person"]["contact"];
//     string cont_phone = contact_node["phone"].as<string>();
//     printf("    phone: %s\n", cont_phone.c_str());
//     string cont_email = contact_node["email"].as<string>();
//     printf("    cont_email: %s\n", cont_email.c_str());

//     std::string string_with_escape = config["string_with_escape"].as<std::string>();
//     std::cout << "String with escape: " << string_with_escape << std::endl;

//     // Access multi-line string
//     std::string string_with_newline = config["string_with_newline"].as<std::string>();
//     std::cout << "Multi-line string:" << std::endl << string_with_newline << std::endl;


//     std::string string_value = config["string"].as<std::string>();
//     int integer_value = config["integer"].as<int>();
//     double float_value = config["float"].as<double>();
//     bool boolean_value = config["boolean"].as<bool>();

//     // Access null value
//     if (config["null"]) {
//         std::cout << "Null value found" << std::endl;
//     } else {
//         std::cout << "Null value not found" << std::endl;
//     }

//     // Access list
//     const YAML::Node& list = config["list"];
//     std::cout << "List values:" << std::endl;
//     for (const auto& item : list) {
//         std::cout << "  " << item.as<int>() << std::endl;
//     }

//     // Access map
//     const YAML::Node& map = config["map"];
//     std::cout << "Map values:" << std::endl;
//     for (const auto& kv : map) {
//         std::cout << "  " << kv.first.as<std::string>() << ": " << kv.second.as<std::string>() << std::endl;
//     }
//     return 0;
// }