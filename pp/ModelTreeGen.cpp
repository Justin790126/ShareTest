#include "ModelTreeGen.h"

ModelTreeGen::ModelTreeGen(QWidget *parent) : QThread(parent) {
    // Initialize model tree and other variables
    
}


void ModelTreeGen::CreateExampleNode()
{
    if (!m_RootNode) {
        m_RootNode = new Node;
        m_RootNode->m_iLyr = 0;
        m_RootNode->m_iShapeType = 0;
        m_RootNode->m_rect.push_back(QRectF(0, 0, 1024, 768));
        m_RootNode->m_children.resize(128);
        
        srand(time(0));

        for (int i = 0; i < 128; i++) {
            Node* node = &m_RootNode->m_children[i];
            node->m_iLyr = i;
            node->m_iShapeType = 0; // 0: rectangle, 1: polygon
            if (node->m_iShapeType == 0) {
                node->m_rect.push_back(QRectF(rand() % 1024, rand() % 768, rand() % 500 + 50, rand() % 300 + 50));
            } else {
                node->m_polygon.push_back(QPolygonF());
                int numVertices = rand() % 10 + 3; // 3-10 vertices
                for (int j = 0; j < numVertices; j++) {
                    node->m_polygon.back() << QPoint(rand() % 1024, rand() % 768);
                }
            }
            // node child
            
        }
        m_RootNode->m_children.shrink_to_fit();
    }
}

void ModelTreeGen::draw() {
    // traverse m_RootNode
    if (!m_RootNode) return;

    if (m_iTgtLyr == 0) {
        // draw white background with rectangle
        QPainter painter(m_pImg);
        painter.fillRect(0, 0, m_pImg->width(), m_pImg->height(), Qt::white);
        
    } else if (m_iTgtLyr == -1) {
        QPainter painter(m_pImg);
        painter.fillRect(0, 0, m_pImg->width(), m_pImg->height(), Qt::white); // Transparent background
        // draw all
        cout << "draw all" << endl;
        for (int i = 0; i < m_RootNode->m_children.size(); i++) {
            painter.setBrush(QColor(rand()%256, 0, 0, 128)); // Semi-transparent Red
            painter.setPen(Qt::black);
            Node* node = &m_RootNode->m_children[i];
            
            if (node->m_iShapeType == 0) {
                painter.drawRect(node->m_rect[0]);
            } else {
                painter.drawPolygon(node->m_polygon[0]);
            }
            
        }

    }else {
        // draw polygons in target layer
        QPainter painter(m_pImg);
        painter.setBrush(QColor(rand()%256, 0, 0, 128)); // Semi-transparent Red
        painter.setPen(Qt::black);
        for (int i = 0; i < m_RootNode->m_children.size(); i++) {
            painter.setBrush(QColor(rand()%256, 0, 0, 128)); // Semi-transparent Red
            painter.setPen(Qt::black);
            Node* node = &m_RootNode->m_children[i];

            if (node->m_iLyr != m_iTgtLyr) {
                continue;
            } else {
                if (node->m_iShapeType == 0) {
                    painter.drawRect(node->m_rect[0]);
                } else {
                    painter.drawPolygon(node->m_polygon[0]);
                }
            }
        }

    }

}

void ModelTreeGen::run() {
    

}

void ModelTreeGen::Wait()
{
    while (this->isRunning()) {
        usleep(1000);
        QApplication::processEvents();
    }
}