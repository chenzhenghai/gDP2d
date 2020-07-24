#pragma once

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/partition.h>
#include <thrust/scan.h>
#include <thrust/remove.h>
#include <thrust/extrema.h>
#include <thrust/unique.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/iterator/reverse_iterator.h>
#include "cudaStructure.h"

struct isEmpty
{
	__device__ bool operator()(const TStatus t)
	{
		return t.isNull();
	}
};

struct isDeleted
{
	__device__ bool operator()(const PStatus x)
	{
		return x.isDeleted();
	}
};

struct isNotDeleted
{
	__device__ bool operator()(const PStatus x)
	{
		return !x.isDeleted();
	}
};


struct isCheckOrBadTriangle
{
	__device__ bool operator()(const TStatus t)
	{
		return (!t.isNull() && (t.isCheck() || t.isBad()));
	}
};

struct isBadTriangle
{
	__device__ bool operator()(const TStatus t)
	{
		return (!t.isNull() && t.isBad());
	}
};

struct isFlipFlop
{
	__device__ bool operator()(const TStatus t)
	{
		return (!t.isNull() && t.isFlip());
	}
};

struct isFlipWinner
{
	__device__ bool operator()(const TStatus t)
	{
		return (!t.isNull() && (t.getFlipOri() != 15));
	}
};

struct isNotNegativeInt
{
	__device__ bool operator()(const int x)
	{
		return !(x < 0);
	}
};

struct isNegativeInt
{
	__device__ bool operator()(const int x)
	{
		return x < 0;
	}
};

struct isNotNegative
{
	__device__ bool operator()(const BYTE x)
	{
		return !(x < 0);
	}
};

struct isTriangleSplit
{
	__device__ bool operator()(const PStatus x)
	{
		return (!x.isDeleted() && x.isSteiner() && !x.isSegmentSplit());
	}
};

struct isRedundant
{
	__device__ bool operator()(const PStatus x)
	{
		return (!x.isDeleted() && x.isReduntant());
	}
};

// Thrust
typedef thrust::device_ptr<int> IntDPtr;
typedef thrust::device_ptr<char> ByteDPtr;
typedef thrust::device_ptr<unsigned long long> UInt64DPtr;
typedef thrust::device_ptr<int2> Int2DPtr;
typedef thrust::device_ptr<REAL2> Real2DPtr;

typedef thrust::device_vector<int> IntD;
typedef thrust::device_vector<int2> Int2D;
typedef thrust::device_vector<BYTE> ByteD;
typedef thrust::device_vector<unsigned long long> UInt64D;
typedef thrust::device_vector<REAL2> Real2D;
typedef thrust::device_vector<REAL> RealD;

typedef thrust::host_vector<int> IntH;
typedef thrust::host_vector<REAL2> Real2H;
typedef thrust::host_vector<REAL> RealH;

typedef thrust::device_vector<PStatus>PStatusD;
typedef thrust::device_ptr<PStatus>PStatusDPtr;
typedef thrust::host_vector<PStatus>PStatusH;

typedef thrust::device_vector<TStatus>TStatusD;
typedef thrust::device_ptr<TStatus>TStatusDPtr;
typedef thrust::host_vector<TStatus>TStatusH;

typedef thrust::device_vector<int2>IntPairD;

typedef thrust::device_vector<unsigned long long>ULongLongD;

int updateEmptyPoints
(
	PStatusD &t_PStatus,
	IntD	  &t_emptypoints
);

int updateEmptyTriangles
(
	TStatusD	&t_TStatus,
	IntD		&t_emptytriangles
);

int updateActiveListByMarker_Slot
(
	IntD	    &t_marker,
	IntD		&t_active,
	int         numberofelements
);

int updateActiveListByMarker_Val
(
	IntD	    &t_marker,
	IntD		&t_active,
	int         numberofelements
);

int updateActiveListByFlipMarker
(
	TStatusD	&t_TStatus,
	IntD		&t_active,
	int         numberoftriangles
);

int updateActiveListByFlipWinner
(
	TStatusD	&t_TStatus,
	IntD		&t_active,
	int         numberoftriangles,
	int			minsize
);

int updateActiveListByFlipNegate
(
	IntD		&t_flipBy,
	IntD		&t_active,
	int         numberoftriangles,
	int			minsize
);
