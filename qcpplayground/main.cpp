// #include <yaml-cpp/yaml.h>
// #include <iostream>
// #include <stdlib.h>

using namespace std;

#include <QApplication>
#include <QWidget>
#include "ViewYmlDisplay.h"

int main(int argc, char *argv[]) {
    QApplication a(argc, argv);
    ViewYmlDisplay* wid = new ViewYmlDisplay;
    wid->resize(640, 480);
    wid->show();
    // Webpage drawingWidget;
    // drawingWidget.resize(400, 300);
    // drawingWidget.show();
    return a.exec();
}

#if WS_TEST
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <cwchar>
#include <codecvt>
#include <locale>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>

using namespace std;

std::vector<std::wstring> split(const std::wstring& str, const std::wstring& delimiters) {
    std::vector<std::wstring> tokens;
    std::wstring token;
    std::wstringstream ss(str);

    while (std::getline(ss, token, delimiters[0])) {
        tokens.push_back(token);
    }

    return tokens;
}

std::vector<std::string> split(const std::string& str, const std::string& delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;

    while (std::getline(ss, token, delimiter[0])) {
        cout << std::stod(token) << endl;
        tokens.push_back(token);
    }

    return tokens;
}

int main() {
    std::wstring str = L"1.234\t999\t123\t3.1415\t9.2622222543\r\n";
    str = str.substr(0, str.find_last_not_of(L"\r\n") + 1);
    std::wstring delimiters = L" \t\r\n,";

    std::vector<std::wstring> tokens = split(str, delimiters);


    // std::cout << str << std::endl;
    
    for (const auto& token : tokens) {
        double val = stod(wstring(token.begin(),token.end()));
        
        cout << val << endl;
    }

    // split_to_double(str, delimiters);
    cout << "convert from wstring to string, and split by delimeter and convert" << endl;

    std::wcout << std::endl;
    std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
    std::string str2 = conv.to_bytes(str);

    vector<string> tks = split(str2, "\t");

    return 0;
}
#endif

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