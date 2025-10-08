#ifndef KD_TREE_H
#define KD_TREE_H

#include <list>
#include "Vector.h"

namespace jmk
{
    class KDTree {
        struct KDRegion2D {
          float left, right, bot, top;
        };
        struct KDNode {
            KDNode* left;
            KDNode* right;
            Vector2f data;
            float value = __FLT_MIN__;
            KDRegion2D boundary{-10, 10, -10, 10};

            KDNode(float _value, KDNode* _left = nullptr, KDNode* _right = nullptr) : value(_value), left(_left), right(_right) {}
            KDNode(Vector2f _data, KDNode* _left = nullptr, KDNode* _right = nullptr) : data(_data), left(_left), right(_right) {}

        };

        bool isALeaf(KDNode* _node) { return !_node->left && !_node->right; }

        KDNode* root = nullptr;

        KDRegion2D default_bound{-10, 10, -10, 10};
        KDRegion2D INVALID_RANGE{0, 0, 0, 0};

        KDNode* constructKDTree(std::list<Vector2f>& _data, uint32_t _depth);
        void traverse(KDNode* _node, std::list<Vector2f>& _list);

        void preprocessBoundaries(KDNode* _node, bool _isEvenDepth);

        bool isInside(const KDRegion2D& r1, const KDRegion2D& r2);
        bool isInRange(const Vector2f& p, const KDRegion2D& r);
        bool isIntersect(const KDRegion2D&, const KDRegion2D&);
        void searchKDTree(KDNode* _node, KDRegion2D _range, std::list<Vector2f>& _list);
        void nearestNeighbor(KDNode* _node, const Vector2f& _search_value, float& _current_distance, bool _even_depth, Vector2f& _current_nn);
        public:
            KDTree() {}

            KDTree(std::list<Vector2f> _data) {
                root = constructKDTree(_data, 0);
                root->boundary = default_bound;
                preprocessBoundaries(root, true);
            }
    };

    KDTree::KDNode* KDTree::constructKDTree(std::list<Vector2f>& _data, uint32_t _depth) {
      // depth even, sort by x
      // depth odd, sort by y  

      auto size = _data.size();
      if (size == 1) {
        return new KDNode(_data.front());
      }

      if (_depth%2==0) {
        _data.sort([](Vector2f a, Vector2f b) {
            return a[X] < b[X];
        });
      } else {
        _data.sort([](Vector2f a, Vector2f b) {
            return a[Y] < b[Y];
        });
      }

      auto mid = size / 2;
      auto mid_ptr = _data.begin();
      std::advance(mid_ptr, mid);
      auto left_list = std::list<Vector2f>(_data.begin(), mid_ptr);
      auto right_list = std::list<Vector2f>(mid_ptr, _data.end());

      auto left_child = constructKDTree(left_list, _depth + 1);
      auto right_child = constructKDTree(right_list, _depth + 1);
      return new KDNode((*mid_ptr)[_depth % 2], left_child, right_child);
    }

    void KDTree::traverse(KDNode* _node, std::list<Vector2f>& _list)
    {
      if (!_node) {
        return;
      }

      traverse(_node->left, _list);
      if (isALeaf(_node)) {
        _list.push_back(_node->data);
      }
      traverse(_node->right, _list);
    }

    void KDTree::preprocessBoundaries(KDNode* _node, bool _isEvenDepth)
    {
      if (!_node || isALeaf(_node)) {
        return;
      }

      if (_isEvenDepth) {
        if (_node->left) {
          _node->left->boundary = _node->boundary;
          _node->left->boundary.right = _node->value;
          preprocessBoundaries(_node->left, !_isEvenDepth);
        }

        if (_node->right) {
          _node->right->boundary = _node->boundary;
          _node->right->boundary.left = _node->value;
          preprocessBoundaries(_node->right, !_isEvenDepth);
        }
      } else {
        if (_node->left) {
          _node->left->boundary = _node->boundary;
          _node->left->boundary.top = _node->value;
          preprocessBoundaries(_node->left, !_isEvenDepth);
        }

        if (_node->right) {
          _node->right->boundary = _node->boundary;
          _node->right->boundary.bot = _node->value;
          preprocessBoundaries(_node->right, !_isEvenDepth);
        }
      }
    }

    bool KDTree::isIntersect(const KDRegion2D& r1, const KDRegion2D& r2)
    {
      if (r1.right < r2.left || r1.left > r2.right) {
        return false;
      }
      if (r1.top < r2.bot || r1.bot > r2.top) {
        return false;
      }
      return true;
    }

    bool KDTree::isInside(const KDRegion2D& r1, const KDRegion2D& r2)
    {
      if (r1.left >= r2.left && r1.right <= r2.right && r1.bot >= r2.bot && r1.top <= r2.top) {
        return true;
      }
      return false;
    }

    void KDTree::searchKDTree(KDNode* _node, KDRegion2D _range, std::list<Vector2f>& _list)
    {
      if (isALeaf(_node)) {
        if (isInRange(_node->data, _range)) {
          _list.push_back(_node->data);
        }
      } else {
        if (_node->left) {
          if (isInside(_node->left->boundary, _range)) {
            traverse(_node->left, _list);
          } else if (isIntersect(_node->left->boundary, _range)) {
            searchKDTree(_node->left, _range, _list);
          }
        }

        if (_node->right) {
          if (isInside(_node->right->boundary, _range)) {
            traverse(_node->right, _list);
          } else if (isIntersect(_node->right->boundary, _range)) {
            searchKDTree(_node->right, _range, _list);
          }
        }
      }
    }

    static double sqrd_distance(const Vector2f& a, const Vector2f& b) {
      return (a[X]-b[X])*(a[X]-b[X]) + (a[Y]-b[Y])*(a[Y]-b[Y]);
    }

    void KDTree::nearestNeighbor(KDNode* _node, const Vector2f& _search_value, float& _current_distance, bool _even_depth, Vector2f& _current_nn)
    {
      if (isALeaf(_node)) {
         auto distance = sqrd_distance(_search_value, _node->data);
         if (distance < _current_distance) {
          _current_distance = distance;
          _current_nn = _node->data;
          return;
         }
      } else {
        auto index = _even_depth ? X : Y;
        auto squre_dis = (_search_value[index] - _node->value) * (_search_value[index] - _node->value);
        if (_search_value[index] < _node->value) {
          nearestNeighbor(_node->left, _search_value, _current_distance, !_even_depth, _current_nn);
          if (squre_dis < _current_distance) {
            nearestNeighbor(_node->right, _search_value, _current_distance, !_even_depth, _current_nn);
          }
        } else {
          nearestNeighbor(_node->right, _search_value, _current_distance, !_even_depth, _current_nn);
          if (squre_dis < _current_distance) {
            nearestNeighbor(_node->left, _search_value, _current_distance, !_even_depth, _current_nn);
          }
        }
      }
    }
    

};

#endif /* KD_TREE_H */