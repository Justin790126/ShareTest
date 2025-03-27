#ifndef CORE_H
#define CORE_H

#include <math.h>

#define TOLERANCE 0.0000001

enum class RELATIVE_POSITION {
    LEFT,
    RIGHT,
    BEHIND, // C, A, B
    BEYOND,  // A, B, C
    BETWEEN, // A, C, B
    ORIGIN, // A=C, B
    DESTINATION // A, B=C
};

static bool isEqualD(double x, double y) {
    return fabs(x-y) < TOLERANCE;
}

#endif /* CORE_H */