#include <iostream>
#include <random>

#include <QPainter>

template<typename T>
class TreeNode {
public:
    T data;
    int x;
    int y;
    int width;
    int height;
    TreeNode<T>* left;
    TreeNode<T>* right;

    TreeNode(T val, int x_val, int y_val, int width_val, int height_val) 
        : data(val), x(x_val), y(y_val), width(width_val), height(height_val), left(nullptr), right(nullptr) {}
};

template<typename T>
class Tree {
private:
    TreeNode<T>* root;

    void traverseInOrderHelper(TreeNode<T>* node, void (*callback)(T, int, int, int, int, QPainter*)) {
        if (node == nullptr) return;
        traverseInOrderHelper(node->left, callback);
        callback(node->data, node->x, node->y, node->width, node->height, painter);
        traverseInOrderHelper(node->right, callback);
    }

public:
    Tree() : root(nullptr) {}

    void insert(T val) {
        int x_val = randomCoord(640); // Assuming 640 as the maximum width of the QWidget
        int y_val = randomCoord(480); // Assuming 480 as the maximum height of the QWidget
        int width_val = randomDimension();
        int height_val = randomDimension();
        root = insertHelper(root, val, x_val, y_val, width_val, height_val);
    }

    void setPainter(QPainter* p) {
        painter = p;
    }

    void traverseInOrder(void (*callback)(T, int, int, int, int, QPainter*)) {
        traverseInOrderHelper(root, callback);
    }

private:
    TreeNode<T>* insertHelper(TreeNode<T>* node, T val, int x_val, int y_val, int width_val, int height_val) {
        if (node == nullptr) {
            return new TreeNode<T>(val, x_val, y_val, width_val, height_val);
        }

        if (val < node->data) {
            node->left = insertHelper(node->left, val, x_val, y_val, width_val, height_val);
        } else if (val > node->data) {
            node->right = insertHelper(node->right, val, x_val, y_val, width_val, height_val);
        }

        return node;
    }

    int randomCoord(int max) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distrib(0, max);
        return distrib(gen);
    }

    int randomDimension() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distrib(20, 100); // Assume width and height between 20 and 100
        return distrib(gen);
    }

    QPainter* painter;
};

template<typename T>
void printCallback(T data, int x, int y, int width, int height, QPainter* painter) {
    // Your painting logic here using the QPainter
    painter->drawText(x, y, QString::number(data));
}
