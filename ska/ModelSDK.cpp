#include "ModelSDK.h"

ModelSDK::ModelSDK(QObject *parent)
    : QThread(parent)
{
    if (!m_clnt) {
        m_clnt = new ModelSktClnt;
    }
}

void ModelSDK::DlClose()
{
    ModelSktMsg msg;
}

void ModelSDK::run()
{
    pid_t child_pid = fork();
    cout << "child_pid: " << child_pid << endl;
    if (child_pid <0) {
        cout << "Failed to fork" << endl;
        return;
    } else if (child_pid == 0) {
        ModelSktSvr svr;
        svr.init();
        svr.start();
        svr.Close();
        cout << "EOF Child process: " << child_pid << endl;
        exit(0);
    } else {
        waitpid(child_pid,NULL,0);
    }
    cout << "EOF SDK svr: " << child_pid << endl;
}