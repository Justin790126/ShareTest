#include "SvrCtMake.h"

void SvrContourMake::handleRequest(ModelSktSvr *svr, PktRes *resFromClnt)
{
    if (resFromClnt->cSyncFlg == 0x00)
    {
        m_pfImg = NULL;
        m_pfImg = svr->readPNGToFloat("lena4096_4096.png", 4096, 4096);
        svr->writeFloatToPNG("lenaqq.png", m_pfImg, 4096, 4096);
        m_sImgSize = 4096 * 4096 * 3 * sizeof(float);

        m_pcImg = new char[m_sImgSize];
        memcpy(m_pcImg, m_pfImg, m_sImgSize);

        size_t sendSize = 0;
        m_vpPktOffsetAndPktSize.clear();
        m_vpPktOffsetAndPktSize.reserve(m_sImgSize / m_sBatchSize);
        while (sendSize < m_sImgSize)
        {

            // send img with batchSize by calling Send
            size_t bytesToSend = m_sBatchSize;
            if (sendSize + m_sBatchSize > m_sImgSize)
            {
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
        char *resmsgpkt = resMsg.createPkt(pktLen, SVR_CONTOUR_MAKE, 0x01, 0x01, 0x00);
        svr->Send(resmsgpkt, pktLen);
        if (resmsgpkt)
            delete[] resmsgpkt;
    }
    else if (resFromClnt->cSyncFlg == 0x01)
    {
        // batch send
        int id = resFromClnt->pktId;
        auto pair = m_vpPktOffsetAndPktSize[id];
        size_t offset = pair.first;
        size_t size = pair.second;
        // printf("client ask for pkt id : %d, offset : %zu, size : %zu\n", resFromClnt->pktId, offset, size);

        char *buf = new char[size];
        memcpy(buf, m_pcImg + offset, size);
        svr->Send(buf, size);
        if (buf)
            delete[] buf;
    }
    else if (resFromClnt->cSyncFlg == 0x02)
    {
        // clear resources
        if (m_pfImg)
            delete[] m_pfImg;
        m_pfImg = NULL;
        if (m_pcImg)
            delete[] m_pcImg;
        m_pcImg = NULL;

        ModelSktMsg resMsg;
        size_t pktLen;
        resMsg.serialize<char>(0x01, pktLen);
        char *resmsgpkt = resMsg.createPkt(pktLen, SVR_CONTOUR_MAKE, 0x01, 0x02, 0x00);
        svr->Send(resmsgpkt, pktLen);
        if (resmsgpkt)
            delete[] resmsgpkt;
    }
}