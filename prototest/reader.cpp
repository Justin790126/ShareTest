#include <iostream>
#include <fstream>
#include <string>
#include "addressbook.pb.h"

using namespace std;

void ListPeople(const tutorial::AddressBook& address_book) {
    for (int i = 0; i < address_book.people_size(); i++) {
        const tutorial::Person& person = address_book.people(i);
        cout << "Person ID: " << person.id() << endl;
        cout << "Name: " << person.name() << endl;
        if (person.has_email()) {
            cout << "Email: " << person.email() << endl;
        }

        for (int j = 0; j < person.phones_size(); j++) {
            const tutorial::Person::PhoneNumber& phone_number = person.phones(j);
            switch (phone_number.type())
            {
                case tutorial::Person::PHONE_TYPE_MOBILE:
                    cout << "  Mobile phone #: ";
                    break;
                case tutorial::Person::PHONE_TYPE_HOME:
                    cout << "  Home phone #: ";
                    break;
                case tutorial::Person::PHONE_TYPE_WORK:
                    cout << "  Work phone #: ";
                    break;
            }
            cout << phone_number.number() << endl;
        }
    }
}

int main(int argc, char* argv[]) {
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    if (argc != 2) {
        cerr << "Usage: addressbook_example <addressbook_file>" << endl;
        return -1;
    }

    tutorial::AddressBook address_book;
    {
        fstream input(argv[1], ios::in | ios::binary);
        if (!address_book.ParseFromIstream(&input)) {
            cerr << "Failed to parse address book" << endl;
            return -1;
        }
    }

    ListPeople(address_book);

    return 0;
}