#include "ModelSktSvr.h"

ModelSktSvr::ModelSktSvr()
{
}

bool ModelSktSvr::init()
{
    bool result = false;
    char resMsg[1024];
    server_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket < 0)
    {
        sprintf(resMsg, "[ModelSktSvr] Error creating socket: %s", strerror(errno));
        m_sStatusMsg = std::move(resMsg);
        return result;
    }

    int opt = 1;
    if (setsockopt(server_socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)))
    {
        sprintf(resMsg, "[ModelSktSvr] Error setup socket: %s", strerror(errno));
        m_sStatusMsg = std::move(resMsg);
        return result;
    }
    // Bind the socket to a port
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = inet_addr(m_sIp.c_str());
    server_addr.sin_port = htons(m_iPort); // Replace with desired port

    if (::bind(server_socket, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0)
    {
        sprintf(resMsg, "[ModelSktSvr] Error binding socket: %s", strerror(errno));
        m_sStatusMsg = std::move(resMsg);
        return result;
    }

    // Listen for incoming connections
    if (listen(server_socket, 30) < 0)
    {
        sprintf(resMsg, "[ModelSktSvr] Error listening on socket: %s", strerror(errno));
        m_sStatusMsg = std::move(resMsg);
        return result;
    }

    sprintf(resMsg, "[ModelSktSvr] Server listening on port %d...", m_iPort);
    m_sStatusMsg = std::move(resMsg);
    result = false;
    return result;
}

bool ModelSktSvr::Accept()
{
    char resMsg[1024];
    client_addr_len = sizeof(client_addr);
    client_socket = accept(server_socket, (struct sockaddr *)&client_addr, &client_addr_len);
    if (client_socket < 0)
    {
        sprintf(resMsg, "[ModelSktSvr] Error accepting connection: %s", strerror(errno));
        m_sStatusMsg = std::move(resMsg);
        return false;
    }
    return true;
}

int ModelSktSvr::DlClose(u_char syncFlg)
{
    size_t resMsgLen = 0;
    size_t pktLen = 0;
    char resMsg[128];
    if (syncFlg == 0x00)
    {
        sprintf(resMsg, "[ModelSktSvr] DlClose successfully");
    }
    else if (syncFlg == 0x01)
    {
        sprintf(resMsg, "[ModelSktSvr] Free DlClose memory successfully");
    }
    resMsgLen = strlen(resMsg) + 1;

    if (syncFlg == 0x00)
    {
        printf("---> dlopen action: %s\n", resMsg);
        ModelSktMsg msg;
        msg.serializeArr<char>(resMsg, resMsgLen, pktLen);
        m_pDlClosePkt = msg.createPkt(pktLen, SVR_DLCLOSE, 0x01, 0x00, 0x00);
        Send(m_pDlClosePkt, pktLen);
    }
    else if (syncFlg == 0x01)
    {
        printf("----> free previous pkt\n");
        if (m_pDlClosePkt)
            delete[] m_pDlClosePkt;
        m_pDlClosePkt = NULL;
    }
    m_sStatusMsg = std::move(resMsg);
}

void ModelSktSvr::ContourMake(u_char flg)
{
    ModelSktMsg msg;
    vector<char *> pkts(10);

    size_t cntSize = (4096 * 4096)+10449;
    int batchSize = 16*1024;
    int resPktNum = ceil(cntSize/(float)batchSize);
    printf("cntSize : %zu , resPktNum = %d\n", cntSize, resPktNum);
    char *pkt = NULL;

    if (flg == 0x00)
    {
        // start send
        // compute resist image
        if (!m_pfCurContour)
            m_pfCurContour = new float[cntSize];
        for (size_t i = 0; i < cntSize; i++)
        {
            m_pfCurContour[i] = i;
        }
        m_dCurContourIdx = 0;
        m_dContourPktId = 0;
        m_iBatchId = 0;

        size_t bytes_sent= 0;
        m_iNumPktSent = 0;
        while (bytes_sent < cntSize) {
            size_t bytes_to_send = batchSize;
            if (bytes_sent + batchSize > cntSize) {
                bytes_to_send = cntSize - bytes_sent;
            }
            // printf("gid = %d with size = %zu\n", m_iNumPktSent, bytes_to_send);
            bytes_sent += bytes_to_send;
            m_iNumPktSent++;
        }

        size_t pktLen;
        msg.serialize<int>(m_iNumPktSent, pktLen);
        msg.serialize<size_t>(cntSize, pktLen);
        pkt = msg.createPkt(pktLen, SVR_CONTOUR_MAKE, 0x01, 0x00, m_dContourPktId);
        // pkts[i] = pkt;
        printf("contour make start\n");
        // Send data to client
        Send(pkt, pktLen);
        if (pkt) delete[] pkt;

        printf("first send--->%f \n", m_dCurContourIdx);
    }
    else if (flg == 0x01)
    {
        size_t bytes_sent = 0;
        // for (int i = 0; i < m_iNumPktSent; i++)
        // {
            int bytes_to_send = batchSize;
            if (bytes_sent + batchSize > cntSize) {
                bytes_to_send = cntSize - bytes_sent;
            }
            bytes_sent += bytes_to_send;

            size_t pktLen;
            msg.serializeArr<float>(m_pfCurContour + m_iBatchId * batchSize, bytes_to_send, pktLen);
            pkt = msg.createPkt(pktLen, SVR_CONTOUR_MAKE, 0x01, 0x00, m_iBatchId);
            // pkts[i] = pkt;
            printf("gid : %d with bytes size %zu\n", m_iBatchId, bytes_to_send);
            // Send data to client
            Send(pkt, pktLen);
            if (pkt) delete[] pkt;

            m_iBatchId++;
        // }

        printf("keep sent--->%zu \n", bytes_sent);
    }

    // for (int i = 0; i < 10; i++) {
    //     float data[1024];
    //     for (int j = 0; j < 1024; j++) {
    //         data[j] = (float)j; // Replace with your desired values
    //     }
    //     size_t pktLen;
    //     msg.serializeArr<float>(data, 1024, pktLen);
    //     char* pkt = msg.createPkt(pktLen, SVR_CONTOUR_MAKE, 0x01, 0x00, i);
    //     pkts[i] = pkt;
    //     printf("contour make : %d \n", i);
    //     // Send data to client
    //     Send(pkt, pktLen);
    // }
    // for (size_t i = 0; i < 10; i++)
    // {
    //     if (pkts[i]) delete[] pkts[i];
    //     pkts[i] = NULL;
    // }
    // pkts.clear();
}
#include <iostream>
#include <fstream>
void ModelSktSvr::fakeImg(char* data, size_t& size)
{
    const std::string fileName = "lena4096_4096.png";

    // Open the file in binary mode
    std::ifstream file(fileName, std::ios::binary | std::ios::ate);

    if (!file.is_open()) {
        std::cerr << "Unable to open file" << std::endl;
        data = NULL;
        return;
    }

    // Get the size of the file
    std::streamsize fileSize = file.tellg();  // Get file size by seeking to the end
    file.seekg(0, std::ios::beg);             // Move back to the beginning of the file

    // Dynamically allocate memory for the file content
    data = new char[fileSize];

    // Read the entire file into the allocated memory
    if (file.read(data, fileSize)) {
        std::cout << "File read successfully, size: " << fileSize << " bytes." << std::endl;
        size = fileSize;
        // Optionally, print out the first few bytes (for debugging)
        for (int i = 0; i < 10 && i < fileSize; ++i) {
            std::cout << static_cast<int>(data[i]) << " ";
        }
        std::cout << std::endl;

    } else {
        std::cerr << "Failed to read the file." << std::endl;
        delete[] data;  // Clean up allocated memory
        data = NULL;
    }

}

void ModelSktSvr::start()
{
    cout << "m_bSvrStop: " << (int)m_bSvrStop << endl;
    while (!m_bSvrStop)
    {
        // cout << "svr listening ..." << endl;
        client_addr_len = sizeof(client_addr);
        if (!Accept())
        {
            cout << " Not accepted" << endl;
            m_bSvrStop = true;
            break;
        }
        #if TEST_SEGMENTATION_FAULT
            int* ptr = nullptr;  // Null pointer
            // Attempting to dereference the null pointer will cause a segmentation fault
            std::cout << *ptr << std::endl;
        #endif

        vector<PktRes> res;
        if (!Receive(res))
        {
            cout << " Receive failed " << endl;
            // m_bSvrStop = true;
            // break;
        }

        for (int i = 0; i < res.size(); i++) {
            PktRes* resFromClnt = &res[i];
            char cmdFrClnt = resFromClnt->cSender;
            switch (cmdFrClnt)
            {
                case SVR_SHUTDOWN: {
                    cout << "svr shutdown" << endl;
                    ModelSktMsg resMsg;
                    string test = "bye!";
                    char* testmsg = new char[sizeof(test) + 1];
                    strcpy(testmsg, test.c_str());
                    size_t resLen;
                    resMsg.serializeArr<char>(testmsg, sizeof(test) + 1, resLen);
                    char* resmsgpkt = resMsg.createPkt(resLen);
                    Send(resmsgpkt, resLen);
                    m_bSvrStop = true;
                    break;
                }
                case SVR_CONTOUR_MAKE: {
                    if (resFromClnt->cSyncFlg == 0x00) {
                        m_pfImg = NULL;
                        m_pfImg = readPNGToFloat("lena4096_4096.png", 4096, 4096);
                        writeFloatToPNG("lenaqq.png", m_pfImg, 4096, 4096);
                        m_sImgSize = 4096 * 4096 * 3 * sizeof(float);

                        m_pcImg = new char[m_sImgSize];
                        memcpy(m_pcImg, m_pfImg, m_sImgSize);

                        size_t sendSize = 0;
                        m_vpPktOffsetAndPktSize.clear();
                        m_vpPktOffsetAndPktSize.reserve(m_sImgSize/m_sBatchSize);
                        while (sendSize < m_sImgSize) {
                            
                            // send img with batchSize by calling Send
                            size_t bytesToSend = m_sBatchSize;
                            if (sendSize + m_sBatchSize > m_sImgSize) {
                                bytesToSend = m_sImgSize - sendSize;
                            }
                            size_t offset = sendSize;
                            auto pair = std::make_pair(offset, bytesToSend);
                            m_vpPktOffsetAndPktSize.emplace_back(pair);

                            sendSize += bytesToSend;
                        }
                        m_vpPktOffsetAndPktSize.shrink_to_fit();

                        // echo back to client to ask client for receive batch file
                        ModelSktMsg resMsg;
                        size_t pktLen;
                        resMsg.serialize<size_t>(m_sImgSize, pktLen);
                        resMsg.serialize<size_t>(m_sBatchSize, pktLen);
                        resMsg.serialize<size_t>(m_vpPktOffsetAndPktSize.size(), pktLen);
                        char* resmsgpkt = resMsg.createPkt(pktLen, SVR_CONTOUR_MAKE, 0x01, 0x01, 0x00);
                        Send(resmsgpkt, pktLen);
                    } else if (resFromClnt->cSyncFlg == 0x01) {
                        // size_t sendSize = 0;
                        int id = resFromClnt->pktId;
                        auto pair = m_vpPktOffsetAndPktSize[id];
                        size_t offset = pair.first;
                        size_t size = pair.second;
                        // printf("client ask for pkt id : %d, offset : %zu, size : %zu\n", resFromClnt->pktId, offset, size);
                        
                        
                        char* buf = new char[size];
                        memcpy(buf, m_pcImg+offset, size);
                        Send(buf, size);
                        if (buf) delete[] buf;

                    }

                    // send bytes
                    
                    break;
                }
            }
        }


        close(client_socket);
    }

    if (m_bSvrStop)
    {
        // send response to client and close
        close(client_socket);
    }

    cout << "EOF ModelSktSvr" << endl;
}

void ModelSktSvr::Close()
{
    close(server_socket);
}