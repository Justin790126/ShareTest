#include <QtGui>
#include <QApplication>
#include <QTableWidget>
#include <QTimer>
#include <QVBoxLayout>
#include <dirent.h> // For directory scanning
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>
#include "ProcessList.h"

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);

    // Default username (current user)
    // struct passwd* pw = getpwuid(getuid());
    // QString defaultUsername = pw ? QString(pw->pw_name) : "just126";
    // QString username = defaultUsername;

    // // Parse command-line options with getopt
    // int opt;
    // while ((opt = getopt(argc, argv, "u:")) != -1) {
    //     switch (opt) {
    //         case 'u':
    //             username = QString(optarg);
    //             break;
    //         default:
    //             fprintf(stderr, "Usage: %s [-u username]\n", argv[0]);
    //             return 1;
    //     }
    // }

    ProcessModel* model = new ProcessModel();
    ProcessView* view = new ProcessView;
    ProcessController* controller = new ProcessController(model, view);

    view->setWindowTitle("Process Monitor");
    view->resize(1100, 600);
    view->show();

    return app.exec();
}
