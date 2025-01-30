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
        m_RootNode->m_rect.push_back(new QRectF(0, 0, 1024, 768));
        m_RootNode->m_children.reserve(10); // ten layers
        for (int i = 0; i < 10; i++) {
            // create new node, each node had random 1000~2000 polygons
            Node* node;
            node = new Node;
            node->m_iLyr = i+1;
            node->m_iShapeType = 1;
            node->m_polygon.reserve(1000+i*100);
            for (int j = 0; j < 1000+i*100; j++) {
                QPolygonF* polygon = new QPolygonF;
                polygon->push_back(QPointF(rand() % 1024, rand() % 768));
            }
            m_RootNode->m_children.push_back(node);
        }
        
    }
}

void ModelTreeGen::run() {
    cout << "ModelTreeGen" << endl;
    // traverse m_RootNode
    if (!m_RootNode) return;

    

}