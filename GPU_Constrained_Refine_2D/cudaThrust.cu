#include "cudaThrust.h"

int updateEmptyPoints
(
	PStatusD &t_PStatus,
	IntD	  &t_emptypoints
)
{
	const int pointListSize = t_PStatus.size();

	t_emptypoints.resize(pointListSize);

	thrust::counting_iterator<int> first(0);
	thrust::counting_iterator<int> last = first + pointListSize;

	t_emptypoints.erase(
		thrust::copy_if(
			first,
			last,
			t_PStatus.begin(),
			t_emptypoints.begin(),
			isDeleted()),
		t_emptypoints.end());

	return t_emptypoints.size();
}

int updateEmptyTriangles
(
	TStatusD	&t_TStatus,
	IntD		&t_emptytriangles
)
{
	const int triangleListSize = t_TStatus.size();

	t_emptytriangles.resize(triangleListSize);

	thrust::counting_iterator<int> first(0);
	thrust::counting_iterator<int> last(triangleListSize);

	t_emptytriangles.erase(
		thrust::copy_if(
			first,
			last,
			t_TStatus.begin(),
			t_emptytriangles.begin(),
			isEmpty()),
		t_emptytriangles.end());

	return t_emptytriangles.size();
}

int updateActiveListByMarker_Slot
(
	IntD	    &t_marker,
	IntD		&t_active,
	int         numberofelements
)
{
	thrust::counting_iterator<int> first(0);
	thrust::counting_iterator<int> last = first + numberofelements;

	t_active.resize(numberofelements);

	t_active.erase(
		thrust::copy_if(
			first,
			last,
			t_marker.begin(),
			t_active.begin(),
			isNotNegativeInt()),
		t_active.end());

	return t_active.size();
}

int updateActiveListByMarker_Val
(
	IntD	    &t_marker,
	IntD		&t_active,
	int         numberofelements
)
{
	t_active.resize(numberofelements);

	t_active.erase(
		thrust::copy_if(
			t_marker.begin(),
			t_marker.begin() + numberofelements,
			t_active.begin(),
			isNotNegativeInt()),
		t_active.end());

	return t_active.size();
}

int updateActiveListByFlipMarker
(
	TStatusD	&t_TStatus,
	IntD		&t_active,
	int         numberoftriangles
)
{
	thrust::counting_iterator<int> first(0);
	thrust::counting_iterator<int> last = first + numberoftriangles;

	t_active.resize(numberoftriangles);

	t_active.erase(
		thrust::copy_if(
			first,
			last,
			t_TStatus.begin(),
			t_active.begin(),
			isFlipFlop()),
		t_active.end());

	return t_active.size();
}

int updateActiveListByFlipWinner
(
	TStatusD	&t_TStatus,
	IntD		&t_active,
	int         numberoftriangles,
	int			minsize
)
{
	thrust::counting_iterator<int> first(0);
	thrust::counting_iterator<int> last = first + numberoftriangles;

	t_active.resize(minsize);

	t_active.erase(
		thrust::copy_if(
			first,
			last,
			t_TStatus.begin(),
			t_active.begin(),
			isFlipWinner()),
		t_active.end());

	return t_active.size();
}

int updateActiveListByFlipNegate
(
	IntD		&t_flipBy,
	IntD		&t_active,
	int         numberoftriangles,
	int			minsize
)
{
	thrust::counting_iterator<int> first(0);
	thrust::counting_iterator<int> last = first + numberoftriangles;

	t_active.resize(minsize);

	t_active.erase(
		thrust::copy_if(
			first,
			last,
			t_flipBy.begin(),
			t_active.begin(),
			isNegativeInt()),
		t_active.end());

	return t_active.size();
}
