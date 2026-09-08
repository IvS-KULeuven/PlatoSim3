
#ifndef QUATERNION_H
#define QUATERNION_H

#include <array>
using namespace std;

typedef array<double, 4> quaternionType;

quaternionType qmul(const quaternionType qa, const quaternionType qb);
quaternionType conj(const quaternionType q);


#endif
