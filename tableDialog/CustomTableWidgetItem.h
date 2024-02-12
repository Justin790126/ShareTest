#include <QTableWidgetItem>

class CustomTableWidgetItem : public QTableWidgetItem {
public:
    CustomTableWidgetItem(const QString& text) : QTableWidgetItem(text) {}

    bool operator<(const QTableWidgetItem &other) const override {
        // Custom sorting criteria: compare items as integers if both are numbers,
        // otherwise compare them as strings
        bool ok1, ok2;
        int value1 = text().toInt(&ok1);
        int value2 = other.text().toInt(&ok2);
        if (ok1 && ok2) {
            return value1 < value2;
        } else {
            return text() < other.text();
        }
    }
};
