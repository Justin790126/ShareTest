#include "ModelTreeGen.h"

ModelTreeGen::ModelTreeGen(QWidget *parent) : QThread(parent)
{
    // Initialize model tree and other variables
}

void ModelTreeGen::CreateExampleNode()
{
    if (!m_RootNode)
    {
        m_RootNode = new Node;
        int level = 0;
        int parentLyrNum = 0;
        CreateExampleNode2(m_RootNode, level, parentLyrNum);
    }
}

void ModelTreeGen::TraverseNode(Node *node, int level, int parentLyrNum, QPainter *painter)
{
    // draw node
    painter->setPen(Qt::black);

    if (level == 0)
    {
        // draw rect
        painter->setBrush(QColor(255, 255, 255, 0));
        painter->drawRect(node->m_rect[0]);
        for (int i = 1; i < node->m_children.size(); i++)
        {
            srand(time(0) + i);
            TraverseNode(&(node->m_children[i]), level + 1, parentLyrNum, painter);
        }
    }
    else if (level == 1)
    {
        
        if (m_iTgtLyr != -1 &&  node->m_iLyr != m_iTgtLyr) {
            // cout << "node->m_iLyr: " << node->m_iLyr << endl;
            return;
        } 
        painter->setBrush(QColor(rand() % 256, rand() % 256, rand() % 256, 128));
        painter->drawRect(node->m_rect[0]);
        for (int i = 1; i < node->m_children.size(); i++)
        {
            srand(time(0) + i);
            TraverseNode(&(node->m_children[i]), level + 1, node->m_iLyr, painter);
        }
    }
    else
    {
        painter->setBrush(QColor(rand() % 256, rand() % 256, rand() % 256, 128));
        if (node->m_iShapeType == 0)
        {
            painter->drawRect(node->m_rect[0]);
        }
        else if (node->m_iShapeType == 1)
        {
            painter->drawPolygon(node->m_polygon[0]);
        }
    }
}

void ModelTreeGen::CreateExampleNode2(Node *node, int level, int parentLyrNum)
{
    if (level == 0)
    {
        m_iSubLyr = 0;
        node->m_iLyr = 0;
        node->m_iShapeType = 0;
        node->m_rect.push_back(QRectF(0, 0, 1024, 768));
        node->m_children.resize(128);
        for (size_t i = 0; i < node->m_children.size(); i++)
        {
            // set random seed with i
            srand(time(0) + i);
            CreateExampleNode2(&(node->m_children[i]), level + 1, parentLyrNum);
        }
    }
    else if (level == 1)
    {
        m_iSubLyr += 1;
        node->m_iLyr = m_iSubLyr;
        node->m_iShapeType = 0;
        node->m_rect.push_back(QRectF(rand() % 1024, rand() % 768, rand() % 500 + 50, rand() % 300 + 50));
        node->m_children.resize(32);
        for (size_t i = 0; i < node->m_children.size(); i++)
        {
            srand(time(0) + i);
            CreateExampleNode2(&(node->m_children[i]), level + 1, node->m_iLyr);
        }
    }
    else
    {

        node->m_iLyr = parentLyrNum;
        srand(time(0) + parentLyrNum);
        node->m_iShapeType = rand() % 2;
        if (node->m_iShapeType == 0)
        {
            node->m_rect.push_back(QRectF(rand() % 1024, rand() % 768, rand() % 500 + 50, rand() % 300 + 50));
        }
        else
        {
            node->m_polygon.push_back(QPolygonF());
            int numVertices = rand() % 10 + 3; // 3-10 vertices
            for (int j = 0; j < numVertices; j++)
            {
                node->m_polygon.back() << QPoint(rand() % 1024, rand() % 768);
            }
        }
    }
}

void ModelTreeGen::draw()
{
    // traverse m_RootNode
    if (!m_RootNode)
        return;

    if (m_iTgtLyr == 0)
    {
        // draw white background with rectangle
        QPainter painter(m_pImg);
        painter.fillRect(0, 0, m_pImg->width(), m_pImg->height(), Qt::white);
    }
    else if (m_iTgtLyr == -1)
    {
        QPainter painter(m_pImg);
        painter.fillRect(0, 0, m_pImg->width(), m_pImg->height(), Qt::white); // Transparent background
        cout << "draw all" << endl;
        int level = 0;
        int parentLyrNum = 0;
        TraverseNode(m_RootNode, level, parentLyrNum, &painter);
    }
    else
    {
        // draw polygons in target layer
        QPainter painter(m_pImg);
        m_pImg->fill(Qt::transparent);
        // draw all
        // cout << "draw target layer: " << m_iTgtLyr << endl;
        int level = 0;
        int parentLyrNum = 0;
        TraverseNode(m_RootNode, level, parentLyrNum, &painter);
    }
}

void ModelTreeGen::run()
{
}

void ModelTreeGen::Wait()
{
    while (this->isRunning())
    {
        usleep(1000);
        QApplication::processEvents();
    }
}