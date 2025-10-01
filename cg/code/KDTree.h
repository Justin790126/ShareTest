#ifndef KD_TREE_H
#define KD_TREE_H

#include <list>
#include "Vector.h"

namespace jmk
{
    class KDTree {
        struct KDNode {
            KDNode* left;
            KDNode* right;
            Vector2f data;
            float value = __FLT_MIN__;

            KDNode(float _value, KDNode* _left = nullptr, KDNode* _right = nullptr) : value(_value), left(_left), right(_right) {}
            KDNode(Vector2f _data, KDNode* _left = nullptr, KDNode* _right = nullptr) : data(_data), left(_left), right(_right) {}

        };

        KDNode* root = nullptr;

        KDNode* constructKDTree(std::list<Vector2f>& _data, uint32_t _depth);
        
        public:
            KDTree() {}

            KDTree(std::list<Vector2f> _data) {
                root = constructKDTree(_data, 0);
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

    

};

#endif /* KD_TREE_H */