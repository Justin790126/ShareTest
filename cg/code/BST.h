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
    BSTNode *parent;

    BSTNode(float _value, BSTNode *_left = nullptr, BSTNode *_right = nullptr,
            BSTNode *_parent = nullptr)
        : value(_value), left(_left), right(_right), parent(_parent) {}
  };

  BSTNode *root = nullptr;

  BSTNode *find(BSTNode *_node, float _value);
  BSTNode *minimumNode(BSTNode *_node);
  BSTNode *maximumNode(BSTNode *_node);
  BSTNode *successorNode(BSTNode *_node);
  BSTNode *predecessorNode(BSTNode *_node);
  BSTNode *splitNode(float _min, float _max);

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
  bool remove(float _value);
  float minimum(BSTNode *_node = nullptr);
  float maximum(BSTNode *_node = nullptr);
  void transplant(BSTNode *u, BSTNode *v);
  bool isAleaf(BSTNode *_node) { return !_node->left && !_node->right; }

  bool successor(float _value, float &_successor);
  bool predecessor(float _value, float &_predecessor);

  void inOrderTravers(BSTNode *, vector<float> &);
  void preOrderTravers(BSTNode *, vector<float> &);
  void postOrderTravers(BSTNode *, vector<float> &);

  void find(const float _min, const float _max, std::vector<float> &_list);
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

BST::BSTNode *BST::splitNode(float _min, float _max) {
  auto v = root;
  while (!isAleaf(v) && (_max <= v->value || _min > v->value)) {
    if (_max <= v->value) {
      v = v->left;
    } else {
      v = v->right;
    }
  }
  return v;
}

void BST::find(const float _min, const float _max, std::vector<float> &_list) {
  auto v_split = splitNode(_min, _max);

  if (isAleaf(v_split)) {
    if (v_split->value >= _min && v_split->value < _max) {
      _list.push_back(v_split->value);
    }
  } else {
    auto v = v_split->left;
    while (!isAleaf(v)) {
      if (_min <= v->value) {
        inOrderTravers(v->right, _list);
        _list.push_back(v->value);
        v = v->left;
      } else {
        v = v->right;
      }
    }

    if (v->value >= _min) {
      _list.push_back(v->value);
    }

    v = v_split->right;
    while (!isAleaf(v)) {
      if (_max >= v->value) {
        inOrderTravers(v->left, _list);
        _list.push_back(v->value);
        v = v->right;
      } else {
        v = v->left;
      }
    }

    if (v->value <= _max) {
      _list.push_back(v->value);
    }
  }
}

BST::BSTNode *BST::find(BSTNode *_node, float _value) {
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

void BST::transplant(BSTNode *u, BSTNode *v) {
  if (!u->parent) {
    root = v;
  } else if (u == u->parent->left) {
    u->parent->left = v;
  } else {
    u->parent->right = v;
  }
  if (v) {
    v->parent = u->parent;
  }
}

bool BST::remove(float _value) {
  BSTNode *current_node = find(root, _value);
  if (current_node) {
    BSTNode *current_left = current_node->left;
    BSTNode *current_right = current_node->right;

    if (isAleaf(current_node)) {
      transplant(current_node, nullptr);
    } else if (!current_left) {
      transplant(current_node, current_right);
    } else if (!current_right) {
      transplant(current_node, current_left);
    } else {
      BSTNode *right_min = minimumNode(current_right);

      if (right_min->parent != current_node) {
        transplant(right_min, right_min->right);
        right_min->right = current_node->right;
        right_min->right->parent = right_min;
      }
      transplant(current_node, right_min);
      right_min->left = current_node->left;
      right_min->left->parent = right_min;
    }
  }
}

bool BST::find(float _value) {
  auto result = find(root, _value);
  if (!result) {
    return false;
  }
  return true;
}

BST::BSTNode *BST::minimumNode(BSTNode *_node) {
  if (!_node) {
    _node = root;
  }

  auto temp = _node;
  while (temp->left) {
    temp = temp->left;
  }
  return temp;
}

float BST::minimum(BSTNode *_node) { return minimumNode(_node)->value; }

BST::BSTNode *BST::maximumNode(BSTNode *_node) {
  if (!_node) {
    _node = root;
  }

  auto temp = _node;
  while (temp->right) {
    temp = temp->right;
  }
  return temp;
}

float BST::maximum(BSTNode *_node) { return maximumNode(_node)->value; }

BST::BSTNode *BST::successorNode(BSTNode *_node) {
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

bool BST::successor(float _value, float &_successor) {
  auto val_ptr = find(root, _value);
  if (!val_ptr)
    return false;
  auto ret = successorNode(val_ptr);
  if (!ret)
    return false;
  _successor = ret->value;
  return true;
}

BST::BSTNode *BST::predecessorNode(BSTNode *_node) {
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