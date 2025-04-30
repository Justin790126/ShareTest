#ifndef CORE_H
#define CORE_H

#include <math.h>

namespace jmk
{

#define TOLERANCE 0.0000001
#define ZERO 0

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

static bool _xor(bool x, bool y) {
    return x^y;
}

static float RadianceToDegrees(float val) {
    return 180*val/3.14159;
}

}

#endif /* CORE_H */