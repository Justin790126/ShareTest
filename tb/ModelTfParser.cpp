#include "ModelTfParser.h"

ModelTfWatcher::ModelTfWatcher()
{
}

ModelTfWatcher::~ModelTfWatcher()
{
    ClearLiveInfo();
    m_bStop = true;
}

vector<string> ModelTfWatcher::GetSubFolder()
{
    Utils* utils = Utils::GetInstance();
    vector<string> folders;
    folders.resize(m_vsSubdirs.size());
    for (size_t i = 0; i < m_vsSubdirs.size(); i++)
    {
        folders[i] = utils->GetBaseName(m_vsSubdirs[i]);
    }
    return folders;
}

void ModelTfWatcher::ClearLiveInfo()
{
    for (size_t i = 0; i < m_vinfoTfFiles.size(); i++)
    {
        if (m_vinfoTfFiles[i])
            delete m_vinfoTfFiles[i];
    }
    m_vinfoTfFiles.clear();
}

void ModelTfWatcher::Wait()
{
    while (isRunning())
    {
        usleep(1000);
        QApplication::processEvents();
    }
}

void ModelTfWatcher::ListSubDir(const string &iDir, vector<string> &oSubDirs)
{
    // use QT to realize
    QDir dir(QString::fromStdString(iDir));
    if (!dir.exists())
    {
        printf("Cannot access directory: %s\n", iDir.c_str());
        return;
    }

    dir.setFilter(QDir::Dirs | QDir::NoDotAndDotDot);
    QFileInfoList list = dir.entryInfoList();
    for (const auto &info : list)
    {
        if (info.isDir())
        {
            oSubDirs.push_back(info.absoluteFilePath().toStdString());
        }
    }
}

void ModelTfWatcher::ListTfFiles(const string &iDir, vector<TfLiveInfo *> &oTfFiles)
{
    // use QT to realize
    QDir dir(QString::fromStdString(iDir));
    if (!dir.exists())
    {
        printf("Cannot access directory: %s\n", iDir.c_str());
        return;
    }

    dir.setFilter(QDir::Files | QDir::NoDotAndDotDot);
    QFileInfoList list = dir.entryInfoList();
    for (const auto &info : list)
    {
        // check file satisfy *.tfevents.*
        if (info.fileName().contains(".tfevents."))
        {
            TfLiveInfo *tfInfo = new TfLiveInfo;
            tfInfo->SetFileName(info.absoluteFilePath().toStdString());
            tfInfo->SetFileSize(info.size());
            tfInfo->SetWatchingPos(0); // reset to 0 when new file added
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
    if (m_iVerbose == 1)
    {
        printf("Parsing log dir %s\n", m_sLogDir.c_str());
        for (const auto &dir : m_vsSubdirs)
        {
            printf("Found subdirectory: %s\n", dir.c_str());
        }
    }
    m_vinfoTfFiles.clear();
    for (const auto &dir : m_vsSubdirs)
    {
        ListTfFiles(dir, m_vinfoTfFiles);
    }

    if (m_iVerbose == 2)
    {
        for (const auto &info : m_vinfoTfFiles)
        {
            cout << *info << endl;
        }
    }
    emit tfFileChanged();

    while (!m_bStop)
    {
        // pooling directory and files and compare m_vinfoTfFiles file size
        // if size change, update m_vinfoTfFiles

        // vector<TfLiveInfo*> curInfos;
        // for (const auto& dir : m_vsSubdirs) {
        //     ListTfFiles(dir, curInfos);
        // }

        // // compare curInfos and m_vinfoTfFiles
        // // if diffrent emit signal change, update m_vinfoTfFiles
        // for (const auto& info : curInfos) {
        //     bool found = false;
        //     for (size_t i = 0; i < m_vinfoTfFiles.size(); i++) {
        //         if (info->GetFileName() == m_vinfoTfFiles[i]->GetFileName()) {
        //             found = true;
        //             if (info->GetFileSize()!= m_vinfoTfFiles[i]->GetFileSize()) {
        //                 emit tfFileChanged();
        //             }
        //             break;
        //         }
        //     }
        //     // if (!found) {
        //     //     emit tfFileAdded(info);
        //     // }
        // }

        usleep(1000000); // 1 second
        QApplication::processEvents();
    }

    // TODO: parse m_vsTfFiles
    cout << "End of Tf file watcher" << endl;
}

/*

    ModelTfParser

 */

ModelTfParser::ModelTfParser()
{
}

ModelTfParser::~ModelTfParser()
{
}

void ModelTfParser::ParseKerasTag(const tensorflow::Summary::Value &value)
{
    if (value.has_tensor())
    {
        const tensorflow::TensorProto &tensor = value.tensor();
        Tensor t;
        if (t.FromProto(tensor))
        {
            if (t.dtype() == DT_STRING)
            {
                // Replace 'auto' with 'Eigen::Tensor<tstring, 1, Eigen::RowMajor>'
                std::vector<std::string> strings;
                Eigen::Tensor<tstring, 1, Eigen::RowMajor> string_tensor = t.flat<tstring>();
                strings.reserve(string_tensor.size());
                for (int i = 0; i < string_tensor.size(); ++i)
                {
                    strings.emplace_back(string_tensor(i));
                }
            }
        }
    }
}

void ModelTfParser::ParseFloatTensor(const tensorflow::Summary::Value &value, QVector<float>& losses)
{
    if (value.has_tensor())
    {
        const tensorflow::TensorProto &tensor = value.tensor();
        Tensor t;
        if (t.FromProto(tensor))
        {
            if (t.dtype() == DT_FLOAT) {
                // Get the number of elements in the tensor
                int num_elements = t.NumElements();
                
                // Access the float data directly
                const float* data = t.flat<float>().data();
                
                // Reserve space in QVector to avoid multiple reallocations
                losses.reserve(num_elements);
                
                // Copy the tensor data into the QVector
                for (int i = 0; i < num_elements; ++i) {
                    losses.append(data[i]);
                }
            }
        }
    }
}

void ModelTfParser::ListEntry(const tensorflow::Event &event)
{
    //   cout << "wall time: " << event.wall_time() << endl;
    //   cout << "step: " << event.step() << endl;
    if (event.has_summary())
    {
        const tensorflow::Summary summary = event.summary();
        for (int i = 0; i < summary.value_size(); i++)
        {
            const tensorflow::Summary::Value &value = summary.value(i);
            // Work with value here
            // For example: value.tag(), value.simple_value(), etc.

            // cout << "Tag: " << value.tag() << ", Value: " << value.simple_value() << endl;
            // print tensor

            if (value.tag() == tagKeras)
            {
                ParseKerasTag(value);
            }
            else if (value.tag() == tagEpochLoss)
            {
                ParseFloatTensor(value, m_qvfEpLoss);
            }
            else if (value.tag() == tagEpochAcc)
            {
                ParseFloatTensor(value, m_qvfEpAcc);
            }
        }
    }
}

void ModelTfParser::run()
{
    m_iVerbose = Utils::m_iVerbose;
    printf("Parse tf file %s\n", m_sFname.c_str());
    m_uCurPos = 0;
    std::uint64_t length;
    std::uint32_t crc;

    if (!m_fp.is_open())
    {
        m_fp.open(m_sFname, std::ios::binary);
        m_fp.seekg(m_uCurPos, std::ios::beg);
    }

    m_uCurPos = m_fp.tellg();

    if (m_fp.peek() == EOF)
    {
        m_fp.close();
    }

    m_qvfEpLoss.clear();
    m_qvfEpAcc.clear();
    while (m_fp.read(reinterpret_cast<char *>(&length), sizeof(std::uint64_t)))
    {
        if (m_fp.eof())
        {
            m_fp.clear();
        }

        m_fp.read(reinterpret_cast<char *>(&crc), sizeof(std::uint32_t));

        std::vector<char> buffer(length);
        m_fp.read(&buffer[0], length);

        tensorflow::Event event;
        if (event.ParseFromString(std::string(buffer.begin(), buffer.end())))
        {
            m_uCurPos = m_fp.tellg();
            ListEntry(event);
        }
        // if (event.ParseFromArray(static_cast<void*>(buffer.data()), length)) {
        //   ListEntry(event);
        // }

        m_fp.read(reinterpret_cast<char *>(&crc), sizeof(std::uint32_t));
    }
    m_fp.close();

    if (m_iVerbose == 2) {
        qDebug() << m_qvfEpLoss << endl;
        qDebug() << m_qvfEpAcc << endl;
    }
}

void ModelTfParser::Wait()
{
    while (isRunning())
    {
        usleep(1000);
        QApplication::processEvents();
    }
}
