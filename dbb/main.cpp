#include <QCoreApplication>
#include <QSqlDatabase>
#include <QSqlQuery>
#include <QSqlError>
#include <QDebug>

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    // Create a QSqlDatabase object
    QSqlDatabase db = QSqlDatabase::addDatabase("QSQLITE");

    // Set the database name
    db.setDatabaseName("mydatabase.db");

    // Open the database connection
    if (!db.open()) {
        qDebug() << "Error: " << db.lastError().text();
        return 1;
    }

    // Create a QSqlQuery object
    QSqlQuery query;

    // Execute a SELECT query to retrieve all rows
    if (!query.exec("SELECT * FROM my_table")) {
        qDebug() << "Error: " << query.lastError().text();
        return 1;
    }

    // Iterate through the results
    while (query.next()) {
        QString col1 = query.value(0).toString();
        int col2 = query.value(1).toInt();
        double col3 = query.value(2).toDouble();
        QByteArray col4 = query.value(3).toByteArray();
        QString col5 = query.value(4).toString();
        int col6 = query.value(5).toInt();
        double col7 = query.value(6).toDouble();
        QByteArray col8 = query.value(7).toByteArray();
        QString col9 = query.value(8).toString();
        int col10 = query.value(9).toInt();

        // Process the retrieved data (e.g., print to console)
        qDebug() << col1 << col2 << col3 << col4 << col5 << col6 << col7 << col8 << col9 << col10;
    }

    // Close the database connection
    db.close();

    return 0;
}