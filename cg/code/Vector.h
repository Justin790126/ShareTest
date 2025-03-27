#ifndef VECTOR_H
#define VECTOR_H

#include <iostream>
#include <array>
#include <type_traits>

#include "Core.h"


using namespace std;

namespace jmk {

#define DIM2 2
#define DIM3 3

#define X 0
#define Y 1
#define Z 2


    template<class coordinate_type, size_t dimension = DIM3>
    class Vector {
        static_assert(std::is_arithmetic<coordinate_type>::value, "Vector class can only store Integer or Floating point values");
        static_assert(dimension >= DIM2, "Vector dimension at least should be 2D");

        std::array<coordinate_type, dimension> coords;


        public:

            friend float dotProduct(const Vector& v1, const Vector& v2) {
                if (v1.coords.size() != v2.coords.size()) {
                    return __FLT_MIN__;
                }
                float product = 0.0;
                for (size_t i = 0; i < v1.coords.size(); ++i) {
                    product += v1.coords[i] * v2.coords[i];
                }
                return product;
            }

            Vector() {}

            Vector(std::array<coordinate_type, dimension> _coords) : coords(_coords) {}

            Vector(coordinate_type _x, coordinate_type _y, coordinate_type _z) : coords({_x, _y, _z}) {}

            Vector(coordinate_type _x, coordinate_type _y) : coords({_x, _y}) {}

            // Equality check
            bool operator==(const Vector<coordinate_type, dimension>&) const;

            // Not equal
            bool operator!=(const Vector<coordinate_type, dimension>&);

            // Addition
            Vector<coordinate_type, dimension> operator+(const Vector<coordinate_type, dimension>&) const;

            // Subtraction
            Vector<coordinate_type, dimension> operator-(const Vector<coordinate_type, dimension>&) const;

            // Less than operator
            bool operator<(const Vector<coordinate_type, dimension>&) const;

            // Greater than operator
            bool operator>(const Vector<coordinate_type, dimension>&) const;

            coordinate_type operator[](int) const;

            void assign(int dim, coordinate_type value);

            float magnitude() const;

            void normalize();

    };

    typedef Vector<float, DIM2> Vector2f;
    typedef Vector<float, DIM3> Vector3f;



    template<class coordinate_type, size_t dimension>
    inline bool Vector<coordinate_type, dimension>::operator == (const Vector<coordinate_type, dimension>& _other) const
    {
        for (size_t i = 0; i < dimension; i++) {
            if (!isEqualD(coords[i], _other.coords[i])) {
                return false;
            }
        }
        return true;
    }

    template<class coordinate_type, size_t dimension>
    inline bool Vector<coordinate_type, dimension>::operator!=(const Vector<coordinate_type, dimension>& _other)
    {
        return !(*this == _other);
    }

    template<class coordinate_type, size_t dimension>
    inline Vector<coordinate_type, dimension> Vector<coordinate_type, dimension>::operator+(const Vector<coordinate_type, dimension>& _other) const
    {
        std::array<coordinate_type, dimension> temp_array;
        for (size_t i = 0; i < dimension; i++) {
            temp_array[i] = coords[i] + _other.coords[i];
        }
        return Vector<coordinate_type, dimension>(temp_array);
    }

    template<class coordinate_type, size_t dimension>
    inline Vector<coordinate_type, dimension> Vector<coordinate_type, dimension>::operator-(const Vector<coordinate_type, dimension>& _other) const
    {
        std::array<coordinate_type, dimension> temp_array;
        for (size_t i = 0; i < dimension; i++) {
            temp_array[i] = coords[i] - _other.coords[i];
        }
        return Vector<coordinate_type, dimension>(temp_array);
    }

    template<class coordinate_type, size_t dimension>
    inline bool Vector<coordinate_type, dimension>::operator<(const Vector<coordinate_type, dimension>& _other) const
    {
        for (size_t i = 0; i < dimension; i++) {
            if (this->coords[i] < _other.coords[i]) {
                return true;
            } else if (this->coords[i] > _other.coords[i]) {
                return false;
            }
        }
        return false;
    }

    template<class coordinate_type, size_t dimension>
    inline bool Vector<coordinate_type, dimension>::operator>(const Vector<coordinate_type, dimension>& _other) const
    {
        // if (*this == _other) return false;
        // return!(*this < _other);
        for (size_t i = 0; i < dimension; i++) {
            if (this->coords[i] > _other.coords[i]) {
                return true;
            } else if (this->coords[i] < _other.coords[i]) {
                return false;
            }
        }
        return false;
    }

    template<class coordinate_type, size_t dimension>
    inline coordinate_type Vector<coordinate_type, dimension>::operator[](int index) const
    {
        if (index >= coords.size()) {
            cout << "Index out of bound \n";
            return coordinate_type();
        }
        return coords[index];
    }

    template<class coordinate_type, size_t dimension>
    inline void Vector<coordinate_type, dimension>::assign(int dim, coordinate_type value)
    {
        if (dim >= coords.size()) {
            cout << "Index out of bound \n";
            return;
        }
        coords[dim] = value;
    }

    template<class coordinate_type, size_t dimension>
    inline float Vector<coordinate_type, dimension>::magnitude() const
    {
        float value = 0.0f;
        for (size_t i = 0; i < dimension; i++) {
            value += pow(coords[i], 2);
        }
        return sqrt(value);
    }

    template<class coordinate_type, size_t dimension>
    inline void Vector<coordinate_type, dimension>::normalize() {
        float mag = magnitude();
        if (mag!= 0) {
            for (size_t i = 0; i < dimension; i++) {
                assign(i, coords[i]/mag);
            }
        }

    }

    float crossProduct2D(Vector2f v1, Vector2f v2);

    Vector3f crossProduct2D(Vector3f v1, Vector3f v2);

}


#endif /* VECTOR_H */