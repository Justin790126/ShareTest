#ifndef BST_H
#define BST_H

#include <iostream>
#include <vector>

using namespace std;

namespace jmk {
class BST {
  struct BSTNode {
    BSTNode *left;
    BSTNode *right;
    float value;

    BSTNode(float _value, BSTNode *_left = nullptr, BSTNode *_right = nullptr)
        : value(_value), left(_left), right(_right) {}
  };

  BSTNode *root = nullptr;

public:
  BST() {}

  BST(vector<float> _values, const unsigned int _index = 0) {
    root = new BSTNode(_values[_index]);

    for (size_t i = 0; i < _values.size(); i++) {
      if (i != _index) {
        insert(_values[i]);
      }
    }
  }

  void insert(float _value);

  void inOrderTravers(BSTNode *, vector<float> &);
  void preOrderTravers(BSTNode *, vector<float> &);
  void postOrderTravers(BSTNode *, vector<float> &);
};

void BST::insert(float _value) {
  if (!root) {
    root = new BSTNode(_value);
    return;
  } else {
    auto temp = root;
    while (true) {
      if (_value < temp->value) {
        if (temp->left) {
          temp = temp->left;
        } else {
          temp->left = new BSTNode(_value);
          break;
        }
      } else {
        if (temp->right) {
          temp = temp->right;
        } else {
          temp->right = new BSTNode(_value);
          break;
        }
      }
    }
  }
}

void BST::preOrderTravers(BSTNode *_node, vector<float> &_list) {
  if (!_node)
    return;

  _list.push_back(_node->value);
  preOrderTravers(_node->left, _list);
  preOrderTravers(_node->right, _list);
}

void BST::postOrderTravers(BSTNode *_node, vector<float> &_list) {
  if (!_node)
    return;

  postOrderTravers(_node->left, _list);
  postOrderTravers(_node->right, _list);
  _list.push_back(_node->value);
}

void BST::inOrderTravers(BSTNode *_node, vector<float> &_list) {
  if (!_node)
    return;

  inOrderTravers(_node->left, _list);
  _list.push_back(_node->value);
  inOrderTravers(_node->right, _list);
}

} // namespace jmk

#endif /* BST_H */