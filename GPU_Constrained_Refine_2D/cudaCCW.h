//****************************************************************************************************
// Adapted to run on CUDA by Cao Thanh Tung
// School of Computing, National University of Singapore. 
// Date: 01/03/2009
//****************************************************************************************************

/*****************************************************************************/
/*                                                                           */
/*  Routines for Arbitrary Precision Floating-point Arithmetic               */
/*  and Fast Robust Geometric Predicates                                     */
/*  (predicates.c)                                                           */
/*                                                                           */
/*  May 18, 1996                                                             */
/*                                                                           */
/*  Placed in the public domain by                                           */
/*  Jonathan Richard Shewchuk                                                */
/*  School of Computer Science                                               */
/*  Carnegie Mellon University                                               */
/*  5000 Forbes Avenue                                                       */
/*  Pittsburgh, Pennsylvania  15213-3891                                     */
/*  jrs@cs.cmu.edu                                                           */
/*                                                                           */
/*  This file contains C implementation of algorithms for exact addition     */
/*    and multiplication of floating-point numbers, and predicates for       */
/*    robustly performing the orientation and incircle tests used in         */
/*    computational geometry.  The algorithms and underlying theory are      */
/*    described in Jonathan Richard Shewchuk.  "Adaptive Precision Floating- */
/*    Point Arithmetic and Fast Robust Geometric Predicates."  Technical     */
/*    Report CMU-CS-96-140, School of Computer Science, Carnegie Mellon      */
/*    University, Pittsburgh, Pennsylvania, May 1996.  (Submitted to         */
/*    Discrete & Computational Geometry.)                                    */
/*                                                                           */
/*  This file, the paper listed above, and other information are available   */
/*    from the Web page http://www.cs.cmu.edu/~quake/robust.html .           */
/*                                                                           */
/*****************************************************************************/
#include <float.h>
#include <cuda_runtime.h>

#define REAL double
#define REAL2 double2

#define INEXACT /* Nothing */

__constant__ REAL constData[13]; 
//REAL cuda_splitter;
//REAL cuda_epsilon;
//REAL cuda_resulterrbound;
//REAL cuda_ccwerrboundA, cuda_ccwerrboundB, cuda_ccwerrboundC;
//REAL cuda_iccerrboundA, cuda_iccerrboundB, cuda_iccerrboundC;
//REAL cuda_o3derrboundA, cuda_o3derrboundB, cuda_o3derrboundC;
//REAL infinity; 

//********* Geometric primitives begin here                           *********
//**                                                                         **
//**                                                                         **
//
//* The adaptive exact arithmetic geometric predicates implemented herein are *
//*   described in detail in my paper, "Adaptive Precision Floating-Point     *
//*   Arithmetic and Fast Robust Geometric Predicates."  See the header for a *
//*   full citation.                                                          *
//
//* Which of the following two methods of finding the absolute values is      *
//*   fastest is compiler-dependent.  A few compilers can inline and optimize *
//*   the fabs() call; but most will incur the overhead of a function call,   *
//*   which is disastrously slow.  A faster way on IEEE machines might be to  *
//*   mask the appropriate bit, but that's difficult to do in C without       *
//*   forcing the value to be stored to memory (rather than be kept in the    *
//*   register to which the optimizer assigned it).                           *


//#define Absolute(a)  ((a) >= 0.0 ? (a) : -(a))
#ifdef SINGLE_PRECISION
#define Absolute(a)  fabsf(a)
#else
	#define Absolute(a)  fabs(a)
#endif

//* Many of the operations are broken up into two pieces, a main part that    *
//*   performs an approximate operation, and a "tail" that computes the       *
//*   roundoff error of that operation.                                       *
//*                                                                           *
//* The operations Fast_Two_Sum(), Fast_Two_Diff(), Two_Sum(), Two_Diff(),    *
//*   Split(), and Two_Product() are all implemented as described in the      *
//*   reference.  Each of these macros requires certain variables to be       *
//*   defined in the calling routine.  The variables `bvirt', `c', `abig',    *
//*   `_i', `_j', `_k', `_l', `_m', and `_n' are declared `INEXACT' because   *
//*   they store the result of an operation that may incur roundoff error.    *
//*   The input parameter `x' (or the highest numbered `x_' parameter) must   *
//*   also be declared `INEXACT'.                                             *

#define Fast_Two_Sum_Tail(a, b, x, y) \
    bvirt = x - a; \
    y = b - bvirt

#define Fast_Two_Sum(a, b, x, y) \
    x = (REAL) (a + b); \
    Fast_Two_Sum_Tail(a, b, x, y)

#define Two_Sum_Tail(a, b, x, y) \
    bvirt = (REAL) (x - a); \
    avirt = x - bvirt; \
    bround = b - bvirt; \
    around = a - avirt; \
    y = around + bround

#define Two_Sum(a, b, x, y) \
    x = (REAL) (a + b); \
    Two_Sum_Tail(a, b, x, y)

#define Two_Diff_Tail(a, b, x, y) \
    bvirt = (REAL) (a - x); \
    avirt = x + bvirt; \
    bround = bvirt - b; \
    around = a - avirt; \
    y = around + bround

#define Two_Diff(a, b, x, y) \
    x = (REAL) (a - b); \
    Two_Diff_Tail(a, b, x, y)

#define Split(a, ahi, alo) \
    c = (REAL) (constData[0]/*cuda_splitter*/ * a); \
    abig = (REAL) (c - a); \
    ahi = c - abig; \
    alo = a - ahi

#define Two_Product_Tail(a, b, x, y) \
    Split(a, ahi, alo); \
    Split(b, bhi, blo); \
    err1 = x - (ahi * bhi); \
    err2 = err1 - (alo * bhi); \
    err3 = err2 - (ahi * blo); \
    y = (alo * blo) - err3

#define Two_Product(a, b, x, y) \
    x = (REAL) (a * b); \
    Two_Product_Tail(a, b, x, y)

// Two_Product_Presplit() is Two_Product() where one of the inputs has       
//   already been split.  Avoids redundant splitting.                        

#define Two_Product_Presplit(a, b, bhi, blo, x, y) \
    x = (REAL) (a * b); \
    Split(a, ahi, alo); \
    err1 = x - (ahi * bhi); \
    err2 = err1 - (alo * bhi); \
    err3 = err2 - (ahi * blo); \
    y = (alo * blo) - err3

// Square() can be done more quickly than Two_Product().                     

#define Square_Tail(a, x, y) \
    Split(a, ahi, alo); \
    err1 = x - (ahi * ahi); \
    err3 = err1 - ((ahi + ahi) * alo); \
    y = (alo * alo) - err3

#define Square(a, x, y) \
    x = (REAL) (a * a); \
    Square_Tail(a, x, y)

// Macros for summing expansions of various fixed lengths.  These are all    
//   unrolled versions of Expansion_Sum().                                   

#define Two_One_Sum(a1, a0, b, x2, x1, x0) \
    Two_Sum(a0, b , _i, x0); \
    Two_Sum(a1, _i, x2, x1)

#define Two_One_Diff(a1, a0, b, x2, x1, x0) \
    Two_Diff(a0, b , _i, x0); \
    Two_Sum( a1, _i, x2, x1)

#define Two_Two_Sum(a1, a0, b1, b0, x3, x2, x1, x0) \
    Two_One_Sum(a1, a0, b0, _j, _0, x0); \
    Two_One_Sum(_j, _0, b1, x3, x2, x1)

#define Two_Two_Diff(a1, a0, b1, b0, x3, x2, x1, x0) \
    Two_One_Diff(a1, a0, b0, _j, _0, x0); \
    Two_One_Diff(_j, _0, b1, x3, x2, x1)

// Macro for multiplying a two-component expansion by a single component.    

#define Two_One_Product(a1, a0, b, x3, x2, x1, x0) \
    Split(b, bhi, blo); \
    Two_Product_Presplit(a0, b, bhi, blo, _i, x0); \
    Two_Product_Presplit(a1, b, bhi, blo, _j, _0); \
    Two_Sum(_i, _0, _k, x1); \
    Fast_Two_Sum(_j, _k, x3, x2)


//*****************************************************************************
//*                                                                           *
//*  cuda_fast_expansion_sum_zeroelim()   Sum two expansions, eliminating zero     *
//*                                  components from the output expansion.    *
//*                                                                           *
//*  Sets h = e + f.  See my Robust Predicates paper for details.             *
//*                                                                           *
//*  If round-to-even is used (as with IEEE 754), maintains the strongly      *
//*  nonoverlapping property.  (That is, if e is strongly nonoverlapping, h   *
//*  will be also.)  Does NOT maintain the nonoverlapping or nonadjacent      *
//*  properties.                                                              *
//*                                                                           *
//*****************************************************************************

__inline__ __device__ __host__ int cuda_fast_expansion_sum_zeroelim(int elen, REAL *e, int flen, REAL *f, REAL *h)
{
    REAL Q;
    INEXACT REAL Qnew;
    INEXACT REAL hh;
    INEXACT REAL bvirt;
    REAL avirt, bround, around;
    int eindex, findex, hindex;
    REAL enow, fnow;

    enow = e[0];
    fnow = f[0];
    eindex = findex = 0;
    if ((fnow > enow) == (fnow > -enow)) {
        Q = enow;
        enow = e[++eindex];
    } else {
        Q = fnow;
        fnow = f[++findex];
    }
    hindex = 0;
    if ((eindex < elen) && (findex < flen)) {
        if ((fnow > enow) == (fnow > -enow)) {
            Fast_Two_Sum(enow, Q, Qnew, hh);
            enow = e[++eindex];
        } else {
            Fast_Two_Sum(fnow, Q, Qnew, hh);
            fnow = f[++findex];
        }
        Q = Qnew;
        if (hh != 0.0) {
            h[hindex++] = hh;
        }
        while ((eindex < elen) && (findex < flen)) {
            if ((fnow > enow) == (fnow > -enow)) {
                Two_Sum(Q, enow, Qnew, hh);
                enow = e[++eindex];
            } else {
                Two_Sum(Q, fnow, Qnew, hh);
                fnow = f[++findex];
            }
            Q = Qnew;
            if (hh != 0.0) {
                h[hindex++] = hh;
            }
        }
    }
    while (eindex < elen) {
        Two_Sum(Q, enow, Qnew, hh);
        enow = e[++eindex];
        Q = Qnew;
        if (hh != 0.0) {
            h[hindex++] = hh;
        }
    }
    while (findex < flen) {
        Two_Sum(Q, fnow, Qnew, hh);
        fnow = f[++findex];
        Q = Qnew;
        if (hh != 0.0) {
            h[hindex++] = hh;
        }
    }
    if ((Q != 0.0) || (hindex == 0)) {
        h[hindex++] = Q;
    }
    return hindex;
}

//*****************************************************************************
//*                                                                           *
//*  cuda_scale_expansion_zeroelim()   Multiply an expansion by a scalar,          *
//*                               eliminating zero components from the        *
//*                               output expansion.                           *
//*                                                                           *
//*  Sets h = be.  See my Robust Predicates paper for details.                *
//*                                                                           *
//*  Maintains the nonoverlapping property.  If round-to-even is used (as     *
//*  with IEEE 754), maintains the strongly nonoverlapping and nonadjacent    *
//*  properties as well.  (That is, if e has one of these properties, so      *
//*  will h.)                                                                 *
//*                                                                           *
//*****************************************************************************

__inline__ __device__ __host__ int cuda_scale_expansion_zeroelim(int elen, REAL *e, REAL b, REAL *h)
{
    INEXACT REAL Q, sum;
    REAL hh;
    INEXACT REAL product1;
    REAL product0;
    int eindex, hindex;
    REAL enow;
    INEXACT REAL bvirt;
    REAL avirt, bround, around;
    INEXACT REAL c;
    INEXACT REAL abig;
    REAL ahi, alo, bhi, blo;
    REAL err1, err2, err3;

    Split(b, bhi, blo);
    Two_Product_Presplit(e[0], b, bhi, blo, Q, hh);
    hindex = 0;
    if (hh != 0) {
        h[hindex++] = hh;
    }
    for (eindex = 1; eindex < elen; eindex++) {
        enow = e[eindex];
        Two_Product_Presplit(enow, b, bhi, blo, product1, product0);
        Two_Sum(Q, product0, sum, hh);
        if (hh != 0) {
            h[hindex++] = hh;
        }
        Fast_Two_Sum(product1, sum, Q, hh);
        if (hh != 0) {
            h[hindex++] = hh;
        }
    }
    if ((Q != 0.0) || (hindex == 0)) {
        h[hindex++] = Q;
    }
    return hindex;
}

//*****************************************************************************
//*                                                                           *
//*  cuda_estimate()   Produce a one-word cuda_estimate of an expansion's value.        *
//*                                                                           *
//*  See my Robust Predicates paper for details.                              *
//*                                                                           *
//*****************************************************************************

__inline__ __device__ __host__ REAL cuda_estimate(int elen, REAL *e)
{
    REAL Q;
    int eindex;

    Q = e[0];
    for (eindex = 1; eindex < elen; eindex++) {
        Q += e[eindex];
    }
    return Q;
}

//*****************************************************************************
//*                                                                           *
//*  counterclockwise()   Return a positive value if the points pa, pb, and   *
//*                       pc occur in counterclockwise order; a negative      *
//*                       value if they occur in clockwise order; and zero    *
//*                       if they are collinear.  The result is also a rough  *
//*                       approximation of twice the signed area of the       *
//*                       triangle defined by the three points.               *
//*                                                                           *
//*  Uses exact arithmetic if necessary to ensure a correct answer.  The      *
//*  result returned is the determinant of a matrix.  This determinant is     *
//*  computed adaptively, in the sense that exact arithmetic is used only to  *
//*  the degree it is needed to ensure that the returned value has the        *
//*  correct sign.  Hence, this function is usually quite fast, but will run  *
//*  more slowly when the input points are collinear or nearly so.            *
//*                                                                           *
//*  See my Robust Predicates paper for details.                              *
//*                                                                           *
//*****************************************************************************
__inline__ __device__ __host__ REAL cuda_ccwadapt(REAL2 pa, REAL2 pb, REAL2 pc, REAL detsum)
{
    INEXACT REAL acx, acy, bcx, bcy;
    REAL acxtail, acytail, bcxtail, bcytail;
    INEXACT REAL detleft, detright;
    REAL detlefttail, detrighttail;
    REAL det, errbound;
    REAL B[4], C1[8], C2[12], D[16];
    INEXACT REAL B3;
    int C1length, C2length, Dlength;
    REAL u[4];
    INEXACT REAL u3;
    INEXACT REAL s1, t1;
    REAL s0, t0;

    INEXACT REAL bvirt;
    REAL avirt, bround, around;
    INEXACT REAL c;
    INEXACT REAL abig;
    REAL ahi, alo, bhi, blo;
    REAL err1, err2, err3;
    INEXACT REAL _i, _j;
    REAL _0;

    acx = (REAL) (pa.x - pc.x);
    bcx = (REAL) (pb.x - pc.x);
    acy = (REAL) (pa.y - pc.y);
    bcy = (REAL) (pb.y - pc.y);

    Two_Product(acx, bcy, detleft, detlefttail);
    Two_Product(acy, bcx, detright, detrighttail);

    Two_Two_Diff(detleft, detlefttail, detright, detrighttail,
        B3, B[2], B[1], B[0]);
    B[3] = B3;

    det = cuda_estimate(4, B);
    errbound = constData[4]/*cuda_ccwerrboundB*/ * detsum;
    if ((det >= errbound) || (-det >= errbound)) {
        return det;
    }

    Two_Diff_Tail(pa.x, pc.x, acx, acxtail);
    Two_Diff_Tail(pb.x, pc.x, bcx, bcxtail);
    Two_Diff_Tail(pa.y, pc.y, acy, acytail);
    Two_Diff_Tail(pb.y, pc.y, bcy, bcytail);

    if ((acxtail == 0.0) && (acytail == 0.0)
        && (bcxtail == 0.0) && (bcytail == 0.0)) {
            return det;
    }

    errbound = constData[5]/*cuda_ccwerrboundC*/ * detsum + constData[2]/*cuda_resulterrbound*/ * Absolute(det);
    det += (acx * bcytail + bcy * acxtail)
        - (acy * bcxtail + bcx * acytail);
    if ((det >= errbound) || (-det >= errbound)) {
        return det;
    }

    Two_Product(acxtail, bcy, s1, s0);
    Two_Product(acytail, bcx, t1, t0);
    Two_Two_Diff(s1, s0, t1, t0, u3, u[2], u[1], u[0]);
    u[3] = u3;
    C1length = cuda_fast_expansion_sum_zeroelim(4, B, 4, u, C1);

    Two_Product(acx, bcytail, s1, s0);
    Two_Product(acy, bcxtail, t1, t0);
    Two_Two_Diff(s1, s0, t1, t0, u3, u[2], u[1], u[0]);
    u[3] = u3;
    C2length = cuda_fast_expansion_sum_zeroelim(C1length, C1, 4, u, C2);

    Two_Product(acxtail, bcytail, s1, s0);
    Two_Product(acytail, bcxtail, t1, t0);
    Two_Two_Diff(s1, s0, t1, t0, u3, u[2], u[1], u[0]);
    u[3] = u3;
    Dlength = cuda_fast_expansion_sum_zeroelim(C2length, C2, 4, u, D);

    return(D[Dlength - 1]);
}

static __device__ __host__ REAL orient2dexact(REAL2 pa, REAL2 pb, REAL2 pc)
{
  INEXACT REAL axby1, axcy1, bxcy1, bxay1, cxay1, cxby1;
  REAL axby0, axcy0, bxcy0, bxay0, cxay0, cxby0;
  REAL aterms[4], bterms[4], cterms[4];
  INEXACT REAL aterms3, bterms3, cterms3;
  REAL v[8], w[12];
  int vlength, wlength;

  INEXACT REAL bvirt;
  REAL avirt, bround, around;
  INEXACT REAL c;
  INEXACT REAL abig;
  REAL ahi, alo, bhi, blo;
  REAL err1, err2, err3;
  INEXACT REAL _i, _j;
  REAL _0;

  Two_Product(pa.x, pb.y, axby1, axby0);
  Two_Product(pa.x, pc.y, axcy1, axcy0);
  Two_Two_Diff(axby1, axby0, axcy1, axcy0,
               aterms3, aterms[2], aterms[1], aterms[0]);
  aterms[3] = aterms3;

  Two_Product(pb.x, pc.y, bxcy1, bxcy0);
  Two_Product(pb.x, pa.y, bxay1, bxay0);
  Two_Two_Diff(bxcy1, bxcy0, bxay1, bxay0,
               bterms3, bterms[2], bterms[1], bterms[0]);
  bterms[3] = bterms3;

  Two_Product(pc.x, pa.y, cxay1, cxay0);
  Two_Product(pc.x, pb.y, cxby1, cxby0);
  Two_Two_Diff(cxay1, cxay0, cxby1, cxby0,
               cterms3, cterms[2], cterms[1], cterms[0]);
  cterms[3] = cterms3;

  vlength = cuda_fast_expansion_sum_zeroelim(4, aterms, 4, bterms, v);
  wlength = cuda_fast_expansion_sum_zeroelim(vlength, v, 4, cterms, w);

  return w[wlength - 1];
}

__inline__ __device__ __host__ REAL cuda_ccw(REAL2 pa, REAL2 pb, REAL2 pc)
{
    REAL detleft, detright, det;
    REAL detsum, errbound;

    detleft = (pa.x - pc.x) * (pb.y - pc.y);
    detright = (pa.y - pc.y) * (pb.x- pc.x);
    det = detleft - detright;

    if (detleft > 0.0) {
        if (detright <= 0.0) {
            return det;
        } else {
            detsum = detleft + detright;
        }
    } else if (detleft < 0.0) {
        if (detright >= 0.0) {
            return det;
        } else {
            detsum = -detleft - detright;
        }
    } else {
        return det;
    }

	//if ( detleft * detright <= 0.0 )
	//	return det; 

	detsum = Absolute(detleft) + Absolute(detright); 

    errbound = constData[3]/*cuda_ccwerrboundA*/ * detsum;
    if ((det >= errbound) || (-det >= errbound)) {
        return det;
    }

    return orient2dexact(pa, pb, pc);
    //return cuda_ccwadapt(pa, pb, pc, detsum);
}

__inline__ __device__ __host__ REAL cuda_fast(REAL2 pa, REAL2 pb, REAL2 pc)
{
    REAL detleft, detright, det;
    REAL detsum, errbound;

    detleft = (pa.x - pc.x) * (pb.y - pc.y);
    detright = (pa.y - pc.y) * (pb.x- pc.x);
    det = detleft - detright;

    //if (detleft > 0.0) {
    //    if (detright <= 0.0) {
    //        return det;
    //    } else {
    //        detsum = detleft + detright;
    //    }
    //} else if (detleft < 0.0) {
    //    if (detright >= 0.0) {
    //        return det;
    //    } else {
    //        detsum = -detleft - detright;
    //    }
    //} else {
    //    return det;
    //}

	//if ( detleft * detright <= 0.0 )
	//	return det; 

	detsum = Absolute(detleft) + Absolute(detright); 

    errbound = constData[3]/*cuda_ccwerrboundA*/ * detsum;
    if ((det >= errbound) || (-det >= errbound)) {
        return det;
    }

    return 0.0;
    //return cuda_ccwadapt(pa, pb, pc, detsum);
}

/*
 *  1 : true
 * -1 : false
 *  0 : not sure 
 */
__inline__ __device__ __host__ REAL cuda_incircleexact(REAL2 pa, REAL2 pb, REAL2 pc, REAL2 pd)
{
  INEXACT REAL axby1, bxcy1, cxdy1, dxay1, axcy1, bxdy1;
  INEXACT REAL bxay1, cxby1, dxcy1, axdy1, cxay1, dxby1;
  REAL axby0, bxcy0, cxdy0, dxay0, axcy0, bxdy0;
  REAL bxay0, cxby0, dxcy0, axdy0, cxay0, dxby0;
  REAL ab[4], bc[4], cd[4], da[4], ac[4], bd[4];
  REAL temp8[8];
  int templen;
  REAL abc[12], bcd[12], cda[12], dab[12];
  int abclen, bcdlen, cdalen, dablen;
  REAL det24x[24], det24y[24], det48x[48], det48y[48];
  int xlen, ylen;
  REAL adet[96], bdet[96], cdet[96], ddet[96];
  int alen, blen, clen, dlen;
  REAL abdet[192], cddet[192];
  int ablen, cdlen;
  REAL deter[384];
  int deterlen;
  int i;

  INEXACT REAL bvirt;
  REAL avirt, bround, around;
  INEXACT REAL c;
  INEXACT REAL abig;
  REAL ahi, alo, bhi, blo;
  REAL err1, err2, err3;
  INEXACT REAL _i, _j;
  REAL _0;

  Two_Product(pa.x, pb.y, axby1, axby0);
  Two_Product(pb.x, pa.y, bxay1, bxay0);
  Two_Two_Diff(axby1, axby0, bxay1, bxay0, ab[3], ab[2], ab[1], ab[0]);

  Two_Product(pb.x, pc.y, bxcy1, bxcy0);
  Two_Product(pc.x, pb.y, cxby1, cxby0);
  Two_Two_Diff(bxcy1, bxcy0, cxby1, cxby0, bc[3], bc[2], bc[1], bc[0]);

  Two_Product(pc.x, pd.y, cxdy1, cxdy0);
  Two_Product(pd.x, pc.y, dxcy1, dxcy0);
  Two_Two_Diff(cxdy1, cxdy0, dxcy1, dxcy0, cd[3], cd[2], cd[1], cd[0]);

  Two_Product(pd.x, pa.y, dxay1, dxay0);
  Two_Product(pa.x, pd.y, axdy1, axdy0);
  Two_Two_Diff(dxay1, dxay0, axdy1, axdy0, da[3], da[2], da[1], da[0]);

  Two_Product(pa.x, pc.y, axcy1, axcy0);
  Two_Product(pc.x, pa.y, cxay1, cxay0);
  Two_Two_Diff(axcy1, axcy0, cxay1, cxay0, ac[3], ac[2], ac[1], ac[0]);

  Two_Product(pb.x, pd.y, bxdy1, bxdy0);
  Two_Product(pd.x, pb.y, dxby1, dxby0);
  Two_Two_Diff(bxdy1, bxdy0, dxby1, dxby0, bd[3], bd[2], bd[1], bd[0]);

  templen = cuda_fast_expansion_sum_zeroelim(4, cd, 4, da, temp8);
  cdalen = cuda_fast_expansion_sum_zeroelim(templen, temp8, 4, ac, cda);
  templen = cuda_fast_expansion_sum_zeroelim(4, da, 4, ab, temp8);
  dablen = cuda_fast_expansion_sum_zeroelim(templen, temp8, 4, bd, dab);
  for (i = 0; i < 4; i++) {
    bd[i] = -bd[i];
    ac[i] = -ac[i];
  }
  templen = cuda_fast_expansion_sum_zeroelim(4, ab, 4, bc, temp8);
  abclen = cuda_fast_expansion_sum_zeroelim(templen, temp8, 4, ac, abc);
  templen = cuda_fast_expansion_sum_zeroelim(4, bc, 4, cd, temp8);
  bcdlen = cuda_fast_expansion_sum_zeroelim(templen, temp8, 4, bd, bcd);

  xlen = cuda_scale_expansion_zeroelim(bcdlen, bcd, pa.x, det24x);
  xlen = cuda_scale_expansion_zeroelim(xlen, det24x, pa.x, det48x);
  ylen = cuda_scale_expansion_zeroelim(bcdlen, bcd, pa.y, det24y);
  ylen = cuda_scale_expansion_zeroelim(ylen, det24y, pa.y, det48y);
  alen = cuda_fast_expansion_sum_zeroelim(xlen, det48x, ylen, det48y, adet);

  xlen = cuda_scale_expansion_zeroelim(cdalen, cda, pb.x, det24x);
  xlen = cuda_scale_expansion_zeroelim(xlen, det24x, -pb.x, det48x);
  ylen = cuda_scale_expansion_zeroelim(cdalen, cda, pb.y, det24y);
  ylen = cuda_scale_expansion_zeroelim(ylen, det24y, -pb.y, det48y);
  blen = cuda_fast_expansion_sum_zeroelim(xlen, det48x, ylen, det48y, bdet);

  xlen = cuda_scale_expansion_zeroelim(dablen, dab, pc.x, det24x);
  xlen = cuda_scale_expansion_zeroelim(xlen, det24x, pc.x, det48x);
  ylen = cuda_scale_expansion_zeroelim(dablen, dab, pc.y, det24y);
  ylen = cuda_scale_expansion_zeroelim(ylen, det24y, pc.y, det48y);
  clen = cuda_fast_expansion_sum_zeroelim(xlen, det48x, ylen, det48y, cdet);

  xlen = cuda_scale_expansion_zeroelim(abclen, abc, pd.x, det24x);
  xlen = cuda_scale_expansion_zeroelim(xlen, det24x, -pd.x, det48x);
  ylen = cuda_scale_expansion_zeroelim(abclen, abc, pd.y, det24y);
  ylen = cuda_scale_expansion_zeroelim(ylen, det24y, -pd.y, det48y);
  dlen = cuda_fast_expansion_sum_zeroelim(xlen, det48x, ylen, det48y, ddet);

  ablen = cuda_fast_expansion_sum_zeroelim(alen, adet, blen, bdet, abdet);
  cdlen = cuda_fast_expansion_sum_zeroelim(clen, cdet, dlen, ddet, cddet);
  deterlen = cuda_fast_expansion_sum_zeroelim(ablen, abdet, cdlen, cddet, deter);

  return deter[deterlen - 1];
}

__inline__ __device__ __host__ REAL cuda_inCircle(REAL2 pa, REAL2 pb, REAL2 pc, REAL2 pd)
{
    REAL adx, bdx, cdx, ady, bdy, cdy;
    REAL bdxcdy, cdxbdy, cdxady, adxcdy, adxbdy, bdxady;
    REAL alift, blift, clift;
    REAL det;
    REAL permanent, errbound;

    adx = pa.x - pd.x;
    bdx = pb.x - pd.x;
    cdx = pc.x - pd.x;
    ady = pa.y - pd.y;
    bdy = pb.y - pd.y;
    cdy = pc.y - pd.y;

    bdxcdy = bdx * cdy;
    cdxbdy = cdx * bdy;
    alift = adx * adx + ady * ady;

    cdxady = cdx * ady;
    adxcdy = adx * cdy;
    blift = bdx * bdx + bdy * bdy;

    adxbdy = adx * bdy;
    bdxady = bdx * ady;
    clift = cdx * cdx + cdy * cdy;

    det = alift * (bdxcdy - cdxbdy)
        + blift * (cdxady - adxcdy)
        + clift * (adxbdy - bdxady);

    permanent = (Absolute(bdxcdy) + Absolute(cdxbdy)) * alift
        + (Absolute(cdxady) + Absolute(adxcdy)) * blift
        + (Absolute(adxbdy) + Absolute(bdxady)) * clift;
    errbound = constData[9] /*cuda_iccerrboundA*/ * permanent;
    if ((det > errbound) || (-det > errbound)) 
		return det; 

    return 0.0; //constData[12] /* INF */;
}



__inline__ __device__ __host__ REAL cuda_inCircle_exact(REAL2 pa, REAL2 pb, REAL2 pc, REAL2 pd)
{
    REAL adx, bdx, cdx, ady, bdy, cdy;
    REAL bdxcdy, cdxbdy, cdxady, adxcdy, adxbdy, bdxady;
    REAL alift, blift, clift;
    REAL det;
    REAL permanent, errbound;

    adx = pa.x - pd.x;
    bdx = pb.x - pd.x;
    cdx = pc.x - pd.x;
    ady = pa.y - pd.y;
    bdy = pb.y - pd.y;
    cdy = pc.y - pd.y;

    bdxcdy = bdx * cdy;
    cdxbdy = cdx * bdy;
    alift = adx * adx + ady * ady;

    cdxady = cdx * ady;
    adxcdy = adx * cdy;
    blift = bdx * bdx + bdy * bdy;

    adxbdy = adx * bdy;
    bdxady = bdx * ady;
    clift = cdx * cdx + cdy * cdy;

    det = alift * (bdxcdy - cdxbdy)
        + blift * (cdxady - adxcdy)
        + clift * (adxbdy - bdxady);

    permanent = (Absolute(bdxcdy) + Absolute(cdxbdy)) * alift
        + (Absolute(cdxady) + Absolute(adxcdy)) * blift
        + (Absolute(adxbdy) + Absolute(bdxady)) * clift;
    errbound = constData[9] /*cuda_iccerrboundA*/ * permanent;
    if ((det > errbound) || (-det > errbound)) {
	#ifdef SINGLE_PRECISION
        return copysignf(1.0f, det);
	#else
		return copysign(1.0, det); 
	#endif
    }

    //return 0.0; //constData[12] /* INF */;	
	return cuda_incircleexact(pa, pb, pc, pd);

}



