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
    BSTNode* parent;

    BSTNode(float _value, BSTNode *_left = nullptr, BSTNode *_right = nullptr,
            BSTNode* _parent = nullptr)
        : value(_value), left(_left), right(_right), parent(_parent) {}
  };

  BSTNode *root = nullptr;

  BSTNode* find(BSTNode* _node, float _value);
  BSTNode* minimumNode(BSTNode* _node);
  BSTNode* maximumNode(BSTNode* _node);
  BSTNode* successorNode(BSTNode* _node);
  BSTNode* predecessorNode(BSTNode* _node);

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
  bool find(float _value);
  float minimum(BSTNode* _node = nullptr);
  float maximum(BSTNode* _node = nullptr);

  bool successor(float _value, float& _successor);
  bool predecessor(float _value, float& _predecessor);

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
          temp->left->parent = temp;
          break;
        }
      } else {
        if (temp->right) {
          temp = temp->right;
        } else {
          temp->right = new BSTNode(_value);
          temp->right->parent = temp;
          break;
        }
      }
    }
  }
}

BST::BSTNode* BST::find(BSTNode* _node, float _value) {
  auto current = _node;
  while (current && current->value != _value) {
    if (current->value < _value) {
      current = current->left;
    } else {
      current = current->right;
    }
  }
  return current;
}

bool BST::find(float _value) {
  auto result = find(root, _value);
  if (!result) {
    return false;
  }
  return true;
}

BST::BSTNode* BST::minimumNode(BSTNode* _node) {
  if (!_node) {
    _node = root;
  }

  auto temp = _node;
  while (temp->left) {
    temp = temp->left;
  }
  return temp;
}

float BST::minimum(BSTNode* _node) {
  return minimumNode(_node)->value;
}

BST::BSTNode* BST::maximumNode(BSTNode* _node) {
  if (!_node) {
    _node = root;
  }

  auto temp = _node;
  while (temp->right) {
    temp = temp->right;
  }
  return temp;
}

float BST::maximum(BSTNode* _node) {
  return maximumNode(_node)->value;
}

BST::BSTNode* BST::successorNode(BSTNode* _node) {
  if (_node->right) {
    return minimumNode(_node->right);
  } else {
    auto temp = _node;
    while (temp->parent && temp == temp->parent->right) {
      temp = temp->parent;
    }
    return temp->parent;
  }
};

bool BST::successor(float _value, float& _successor) 
{
  auto val_ptr = find(root, _value);
  if (!val_ptr) return false;
  auto ret = successorNode(val_ptr);
  if (!ret) return false;
  _successor = ret->value;
  return true;
}

BST::BSTNode* BST::predecessorNode(BSTNode* _node) {
  if (_node->left) {
    return maximumNode(_node->left);
  } else {
    auto temp = _node;
    while (temp->parent && temp == temp->parent->left) {
      temp = temp->parent;
    }
    return temp->parent;
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