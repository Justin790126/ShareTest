#ifndef MODEL_TREE_GEN_H
#define MODEL_TREE_GEN_H

#include <iostream>
#include <vector>
#include <string>
#include <QtGui>

using namespace std;

enum ShapeType
{
    SHAPE_RECT = 0,
    SHAPE_POLYGON
};

struct Node
{
    int m_iLyr;
    int m_iShapeType;
    vector<QRectF*> m_rect;
    vector<QPolygonF*> m_polygon;
    vector<Node*> m_children;
};

class ModelTreeGen : public QThread
{
    Q_OBJECT
    public:
    ModelTreeGen(QWidget *parent=NULL);
    ~ModelTreeGen() = default;

    void CreateExampleNode();

    Node* GetRootNode() const { return m_RootNode; }
    void SetRootNode(Node* pRoot) { m_RootNode = pRoot; }
    void SetImage(QImage* pImg) { m_pImg = pImg; }
    void SetTargetLyr(int lyr) { m_iTgtLyr = lyr; }

    void run() override;

    private:
        Node* m_RootNode;

        QImage* m_pImg;

        int m_iTgtLyr;
};


#endif /* MODEL_TREE_GEN_H */