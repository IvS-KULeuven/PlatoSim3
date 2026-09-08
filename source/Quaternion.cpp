
#include "Quaternion.h"


/**
  * \brief  Multiplication of two quaternions qa and qb.
  * 
  * \input Both quaternions are structured as (q0, qx, qy, qz)
  *
  * \output A quaternion with the same structure.
  */ 
quaternionType qmul(const quaternionType qa, const quaternionType qb)
{
    quaternionType result;
    result[0] = qa[0]*qb[0] - qa[1]*qb[1] - qa[2]*qb[2] - qa[3]*qb[3];
    result[1] = qa[0]*qb[1] + qa[1]*qb[0] + qa[2]*qb[3] - qa[3]*qb[2];
    result[2] = qa[0]*qb[2] - qa[1]*qb[3] + qa[2]*qb[0] + qa[3]*qb[1];
    result[3] = qa[0]*qb[3] + qa[1]*qb[2] - qa[2]*qb[1] + qa[3]*qb[0];
    return result;
}






/**
  * \brief  Conjugate of a quaternion 
  * 
  * \input Quaternions is structured as (q0, qx, qy, qz)
  *
  * \output A quaternion with the same structure.
  */ 
quaternionType conj(const quaternionType q) 
{
    quaternionType result;
    result[0] = q[0];
    result[1] = -q[1];
    result[2] = -q[2];
    result[3] = -q[3];
    return result;
}
