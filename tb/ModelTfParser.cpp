#include "ModelTfParser.h"

ModelTfWatcher::ModelTfWatcher()
{

}

ModelTfWatcher::~ModelTfWatcher()
{
    ClearLiveInfo();
}

void ModelTfWatcher::ClearLiveInfo()
{
    for (size_t i = 0; i < m_vsTfFilesMonitorPos.size(); i++) {
        if (m_vsTfFilesMonitorPos[i]) delete m_vsTfFilesMonitorPos[i];
    }
    m_vsTfFilesMonitorPos.clear();
}

void ModelTfWatcher::Wait()
{
    while (isRunning())
    {
        usleep(1000);
        QApplication::processEvents();
    }
}

void ModelTfWatcher::ListSubDir(const string & iDir, vector<string>& oSubDirs)
{
    // use QT to realize
    QDir dir(QString::fromStdString(iDir));
    if (!dir.exists()) {
        printf("Cannot access directory: %s\n", iDir.c_str());
        return;
    }

    dir.setFilter(QDir::Dirs | QDir::NoDotAndDotDot);
    QFileInfoList list = dir.entryInfoList();
    for (const auto& info : list) {
        if (info.isDir()) {
            oSubDirs.push_back(info.absoluteFilePath().toStdString());
        }
    }
}

void ModelTfWatcher::ListTfFiles(const string & iDir, vector<TfLiveInfo*>& oTfFiles)
{
    // use QT to realize
    QDir dir(QString::fromStdString(iDir));
    if (!dir.exists()) {
        printf("Cannot access directory: %s\n", iDir.c_str());
        return;
    }

    dir.setFilter(QDir::Files | QDir::NoDotAndDotDot);
    QFileInfoList list = dir.entryInfoList();
    for (const auto& info : list) {
        // check file satisfy *.tfevents.*
        if (info.fileName().contains(".tfevents.")) {
            TfLiveInfo* tfInfo = new TfLiveInfo;
            tfInfo->SetFileName(info.fileName().toStdString());
            tfInfo->SetFileSize(info.size());
            tfInfo->SetWatchingPos(0);  // reset to 0 when new file added
            oTfFiles.push_back(tfInfo);
        }
    }

    // monitor file change
}


void ModelTfWatcher::run()
{
    m_iVerbose = Utils::m_iVerbose;
    m_vsSubdirs.clear();
    ListSubDir(m_sLogDir, m_vsSubdirs);
    if (m_iVerbose == 1) {
        printf("Parsing log dir %s\n", m_sLogDir.c_str());
        for (const auto& dir : m_vsSubdirs) {
            printf("Found subdirectory: %s\n", dir.c_str());
        }
    }
    m_vsTfFilesMonitorPos.clear();
    for (const auto& dir : m_vsSubdirs) {
        ListTfFiles(dir, m_vsTfFilesMonitorPos);
    }
    
    if (m_iVerbose == 2) {
        printf("Found TensorFlow event files in %s\n", m_sLogDir.c_str());
        for (const auto& info : m_vsTfFilesMonitorPos) {
            cout << info << endl;
        }
    }
    emit tfFileChanged();

    while (!m_bStop) {
        // pooling directory and files and compare m_vsTfFilesMonitorPos file size
        // if size change, update m_vsTfFilesMonitorPos

        // vector<TfLiveInfo*> curInfos;
        // for (const auto& dir : m_vsSubdirs) {
        //     ListTfFiles(dir, curInfos);
        // }

        // // compare curInfos and m_vsTfFilesMonitorPos
        // // if diffrent emit signal change, update m_vsTfFilesMonitorPos
        // for (const auto& info : curInfos) {
        //     bool found = false;
        //     for (size_t i = 0; i < m_vsTfFilesMonitorPos.size(); i++) {
        //         if (info->GetFileName() == m_vsTfFilesMonitorPos[i]->GetFileName()) {
        //             found = true;
        //             if (info->GetFileSize()!= m_vsTfFilesMonitorPos[i]->GetFileSize()) {
        //                 emit tfFileChanged();
        //             }
        //             break;
        //         }
        //     }
        //     // if (!found) {
        //     //     emit tfFileAdded(info);
        //     // }
        // }

        usleep(1000000);  // 1 second
        QApplication::processEvents();

    }

    // TODO: parse m_vsTfFiles
    cout << "End of Tf file watcher" << endl;
}