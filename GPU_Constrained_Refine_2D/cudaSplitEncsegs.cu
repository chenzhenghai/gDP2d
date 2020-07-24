#include "cudaSplitEncsegs.h"
#include "cudaFlipFlop.h"
#include "cudaMesh.h"

void splitEncsegs(
	Real2D	 &t_pointlist,
	PStatusD &t_PStatus,
	IntD	 &t_trianglelist,
	IntD	 &t_neighborlist,
	IntD	 &t_tri2subseg,
	TStatusD &t_TStatus,
	IntD	 &t_segmentlist,
	IntD	 &t_subseg2tri,
	IntD	 &t_subseg2seg,
	IntD	 &t_encmarker,
	IntD	 &t_enclist,
	IntD	 &t_internalmarker,
	IntD	 &t_internallist,
	IntD	 &t_flipBy,
	IntD	 &t_flipActive,
	IntD	 &t_linklist,
	IntD	 &t_linkslot,
	IntD	 &t_emptypoints,
	IntD	 &t_emptytriangles,
	int pointblock,
	int triblock,
	int * numberofemptypoints,
	int * numberofemptytriangles,
	int * numberofpoints,
	int	* numberoftriangles,
	int	* numberofsubseg,
	int encmode,
	REAL theta,
	int debug_iter
)
{
	int numberofencs; // number of encroached subsegs
	int numberofdels; // number of subsegs that need to delete their apex
	int numberofmids; // number of subsegs that need to be inserted midpoint

	int numberofblocks;

	int iteration = 0;
	// loop until there is no encroached subseg left
	while (true)
	{
		// update encroached subsegs active list
		numberofencs = updateActiveListByMarker_Slot(t_encmarker, t_enclist, *numberofsubseg);

#ifdef GQM2D_DEBUG_2
		printf("Iteration = %d, number of encroached segments = %d\n", iteration, numberofencs);
		if (false)
		{
			int * debug_el = new int[numberofencs];
			cudaMemcpy(debug_el, thrust::raw_pointer_cast(&t_enclist[0]), sizeof(int)*numberofencs, cudaMemcpyDeviceToHost);
			for (int i = 0; i < numberofencs; i++)
				printf("%d ", debug_el[i]);
			printf("\n");
			delete[] debug_el;
		}
#endif

		if (numberofencs == 0)
			break;

		// use internal marker and list for deletion
		// init deletion marker
		t_internalmarker.resize(numberofencs);
		thrust::copy(t_enclist.begin(), t_enclist.end(), t_internalmarker.begin());

		// delete all points inside diametral circle
		int step = 0;
		while (true)
		{
			// update deletion subsegs active list and marker
			numberofdels = updateActiveListByMarker_Val(t_internalmarker, t_internallist, t_internalmarker.size());
#ifdef GQM2D_DEBUG_2
			printf(" numberofdels = %d\n", numberofdels);
#endif

			if (numberofdels == 0)
				break;

			t_internalmarker.resize(numberofdels);

			// mark reduntant points
			markReduntantPoints(
				t_pointlist,
				t_PStatus,
				t_trianglelist,
				t_neighborlist,
				t_TStatus,
				t_subseg2tri,
				t_internalmarker,
				t_internallist,
				numberofdels);

#ifdef GQM2D_DEBUG
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
#endif

			// maintain denauly property, do flip-flop
			flipFlop(
				t_pointlist,
				t_PStatus,
				t_trianglelist,
				t_neighborlist,
				t_tri2subseg,
				t_TStatus,
				t_subseg2tri,
				t_flipBy,
				t_flipActive,
				t_encmarker,
				t_linklist,
				t_linkslot,
				*numberoftriangles,
				encmode,
				theta,
				-1,
				-1);

			// check if encroachment markers are updated correctly
#ifdef GQM2D_DEBUG_3
			{
				printf(" Iteration %d, Step %d: After Remove redundant points\n", iteration, step);
				int * debug_em = new int[*numberofsubseg];
				int * debug_tl = new int[3 * (*numberoftriangles)];
				int * debug_nl = new int[3 * (*numberoftriangles)];
				REAL2 * debug_pl = new REAL2[*numberofpoints];
				int * debug_st = new int[*numberofsubseg];
				cudaMemcpy(debug_em, thrust::raw_pointer_cast(&t_encmarker[0]), sizeof(int)*(*numberofsubseg), cudaMemcpyDeviceToHost);
				cudaMemcpy(debug_tl, thrust::raw_pointer_cast(&t_trianglelist[0]), sizeof(int) * 3 * *numberoftriangles, cudaMemcpyDeviceToHost);
				cudaMemcpy(debug_nl, thrust::raw_pointer_cast(&t_neighborlist[0]), sizeof(int) * 3 * *numberoftriangles, cudaMemcpyDeviceToHost);
				cudaMemcpy(debug_pl, thrust::raw_pointer_cast(&t_pointlist[0]), sizeof(REAL2)**numberofpoints, cudaMemcpyDeviceToHost);
				cudaMemcpy(debug_st, thrust::raw_pointer_cast(&t_subseg2tri[0]), sizeof(int)**numberofsubseg, cudaMemcpyDeviceToHost);

				for (int i = 0; i < *numberofsubseg; i++)
				{
					int otri = debug_st[i];
					int tri = otri >> 2;
					int ori = otri & 3;
					int p[3];
					REAL2 v[3];
					p[0] = debug_tl[3 * tri + (ori + 1) % 3];
					p[1] = debug_tl[3 * tri + (ori + 2) % 3];
					p[2] = debug_tl[3 * tri + ori];
					v[0] = debug_pl[p[0]];
					v[1] = debug_pl[p[1]];
					v[2] = debug_pl[p[2]];
					bool tag = false; // indicate if this segment is encroached or not

					REAL goodcoss = cos(theta * PI / 180.0);
					goodcoss *= goodcoss;
					REAL dotproduct = (v[0].x - v[2].x)*(v[1].x - v[2].x) +
						(v[0].y - v[2].y)*(v[1].y - v[2].y);

					if (dotproduct < 0.0) // angle > 90
					{
						// here, we use diametral lens to speedup the algorithm
						if (encmode || dotproduct * dotproduct >=
							(2.0*goodcoss - 1.0)*(2.0*goodcoss - 1.0) *
							((v[0].x - v[2].x)*(v[0].x - v[2].x) + (v[0].y - v[2].y)*(v[0].y - v[2].y)) *
							((v[1].x - v[2].x)*(v[1].x - v[2].x) + (v[1].y - v[2].y)*(v[1].y - v[2].y)))
							tag = true;
					}

					otri = debug_nl[3 * tri + ori];
					if (otri != -1)
					{
						tri = otri >> 2;
						ori = otri & 3;
						p[2] = debug_tl[3 * tri + ori];
						v[2] = debug_pl[p[2]];
						dotproduct = (v[0].x - v[2].x)*(v[1].x - v[2].x) +
							(v[0].y - v[2].y)*(v[1].y - v[2].y);
						if (dotproduct < 0.0) // angle > 90
						{
							// here, we use diametral lens to speedup the algorithm
							if (encmode || dotproduct * dotproduct >=
								(2.0*goodcoss - 1.0)*(2.0*goodcoss - 1.0) *
								((v[0].x - v[2].x)*(v[0].x - v[2].x) + (v[0].y - v[2].y)*(v[0].y - v[2].y)) *
								((v[1].x - v[2].x)*(v[1].x - v[2].x) + (v[1].y - v[2].y)*(v[1].y - v[2].y)))
								tag = true;
						}
					}

					if (debug_em[i] == -1 && tag)
						printf(" iteration = %d, step = %d, Segment %d: I am encroached but marked as non-encroached\n", iteration, step, i);

					if (debug_em[i] == 0 && !tag)
						printf(" iteration = %d, step = %d, Segment %d: I am not encroached but marked as encroached\n", iteration, step, i);

					//if( debug_em[i] == 1)
					//	printf("Line 3362, iteration = %d, step = %d, Segment %d: I am marked as encroached because I am on segment\n",iteration,step,i);

				}
				delete[] debug_em;
				delete[] debug_tl;
				delete[] debug_nl;
				delete[] debug_pl;
				delete[] debug_st;
				printf(" Finished Checking\n");
			}
#endif

			step++;
		}

#ifdef GQM2D_DEBUG_3
		{
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
			printf(" Iteration %d: After Remove all redundant points\n", iteration);
			int * debug_tl = new int[3 * (*numberoftriangles)];
			REAL2 * debug_pl = new REAL2[*numberofpoints];
			TStatus * debug_ts = new TStatus[*numberoftriangles];
			cudaMemcpy(debug_tl, thrust::raw_pointer_cast(&t_trianglelist[0]), sizeof(int) * 3 * *numberoftriangles, cudaMemcpyDeviceToHost);
			cudaMemcpy(debug_pl, thrust::raw_pointer_cast(&t_pointlist[0]), sizeof(REAL2)**numberofpoints, cudaMemcpyDeviceToHost);
			cudaMemcpy(debug_ts, thrust::raw_pointer_cast(&t_TStatus[0]), sizeof(TStatus)**numberoftriangles, cudaMemcpyDeviceToHost);
			for (int i = 0; i < *numberoftriangles; i++)
			{
				if (!debug_ts[i].isNull())
				{
					bool errorflag = false;
					int p[3];
					REAL2 v[3];
					for (int j = 0; j < 3; j++)
					{
						p[j] = debug_tl[3 * i + j];
						v[j] = debug_pl[p[j]];
					}
					for (int j = 0; j < 2; j++)
					{
						for (int k = j + 1; k < 3; k++)
						{
							if (v[j].x == v[k].x && v[j].y == v[k].y)
							{
								errorflag = true;
							}
						}
					}
					if (errorflag)
						printf("  After remove redundant points - Tri %d: Duplicate vertice\n", i);
				}
			}
			delete[] debug_tl;
			delete[] debug_pl;
			delete[] debug_ts;
			printf(" Finished Checking\n");
		}
#endif

		// check if there is enough space
		// numberofencs points are going to be inserted
		if (numberofencs > *numberofemptypoints)
		{
			*numberofemptypoints = updateEmptyPoints(t_PStatus, t_emptypoints);
			int num = 0;
			while (numberofencs > *numberofemptypoints + num*pointblock)
				num++;
			if (num != 0)
			{
				int old_size = t_PStatus.size();
				PStatus emptyPoint;
				emptyPoint.setDeleted();
				t_pointlist.resize(old_size + num*pointblock);
				t_PStatus.resize(old_size + num*pointblock, emptyPoint);
				*numberofemptypoints = updateEmptyPoints(t_PStatus, t_emptypoints);
			}
		}

		if (2 * numberofencs > *numberofemptytriangles)
		{
			*numberofemptytriangles = updateEmptyTriangles(t_TStatus, t_emptytriangles);
			int num = 0;
			while (2 * numberofencs > *numberofemptytriangles + num*triblock)
				num++;
			if (num != 0)
			{
				int old_size = t_TStatus.size();
				TStatus emptyTri(true, false, false);
				t_trianglelist.resize(3 * (old_size + num*triblock));
				t_neighborlist.resize(3 * (old_size + num*triblock));
				t_tri2subseg.resize(3 * (old_size + num*triblock), -1);
				t_TStatus.resize(old_size + num*triblock, emptyTri);
				t_flipBy.resize(old_size + num*triblock);
				*numberofemptytriangles = updateEmptyTriangles(t_TStatus, t_emptytriangles);
			}
		}

		t_subseg2tri.resize(*numberofsubseg + numberofencs);
		t_subseg2seg.resize(*numberofsubseg + numberofencs);

		// use internal marker and list for insertion subsegs, use t_flipBy as insertion marker
		// init insertion subseg marker
		t_encmarker.resize(*numberofsubseg + numberofencs, -1);
		t_internalmarker.resize(numberofencs);
		thrust::copy(t_enclist.begin(), t_enclist.end(), t_internalmarker.begin());

		// split all encroached subsegs
		while (true)
		{
			// inside one triangle, more than one segment may split, violation may happen

			// update insertion subsegs active list and marker
			// t_internallist store the indices for t_enclist
			// in order to keep thread id for kernels, do not resize t_internalmarker
			numberofmids = updateActiveListByMarker_Slot(t_internalmarker, t_internallist, numberofencs);
#ifdef GQM2D_DEBUG_2
			printf(" numberofmids = %d\n", numberofmids);
#endif

			if (numberofmids == 0)
				break;

			// reset insertion (triangles) marker: t_flipBy and t_flipOri
			numberofblocks = (ceil)((float)numberofmids / BLOCK_SIZE);
			kernelResetMidInsertionMarker << <numberofblocks, BLOCK_SIZE >> >(
				thrust::raw_pointer_cast(&t_neighborlist[0]),
				thrust::raw_pointer_cast(&t_TStatus[0]),
				thrust::raw_pointer_cast(&t_subseg2tri[0]),
				thrust::raw_pointer_cast(&t_flipBy[0]),
				thrust::raw_pointer_cast(&t_internalmarker[0]),
				thrust::raw_pointer_cast(&t_internallist[0]),
				numberofmids);

#ifdef GQM2D_DEBUG
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
#endif

			// mark insertion triangles
			numberofblocks = (ceil)((float)numberofmids / BLOCK_SIZE);
			kernelMarkMidInsertion << <numberofblocks, BLOCK_SIZE >> >(
				thrust::raw_pointer_cast(&t_neighborlist[0]),
				thrust::raw_pointer_cast(&t_TStatus[0]),
				thrust::raw_pointer_cast(&t_subseg2tri[0]),
				thrust::raw_pointer_cast(&t_flipBy[0]),
				thrust::raw_pointer_cast(&t_internalmarker[0]),
				thrust::raw_pointer_cast(&t_internallist[0]),
				numberofmids,
				thrust::raw_pointer_cast(&t_trianglelist[0]),
				thrust::raw_pointer_cast(&t_tri2subseg[0]),
				debug_iter);

#ifdef GQM2D_DEBUG
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
#endif

			// insert points for winners
			numberofblocks = (ceil)((float)numberofmids / BLOCK_SIZE);
			kernelInsertMidPoints << <numberofblocks, BLOCK_SIZE >> >(
				thrust::raw_pointer_cast(&t_pointlist[0]),
				thrust::raw_pointer_cast(&t_PStatus[0]),
				thrust::raw_pointer_cast(&t_trianglelist[0]),
				thrust::raw_pointer_cast(&t_neighborlist[0]),
				thrust::raw_pointer_cast(&t_tri2subseg[0]),
				thrust::raw_pointer_cast(&t_TStatus[0]),
				thrust::raw_pointer_cast(&t_segmentlist[0]),
				thrust::raw_pointer_cast(&t_subseg2tri[0]),
				thrust::raw_pointer_cast(&t_subseg2seg[0]),
				thrust::raw_pointer_cast(&t_encmarker[0]),
				thrust::raw_pointer_cast(&t_internalmarker[0]),
				thrust::raw_pointer_cast(&t_internallist[0]),
				thrust::raw_pointer_cast(&t_flipBy[0]),
				thrust::raw_pointer_cast(&t_emptypoints[0]),
				thrust::raw_pointer_cast(&t_emptytriangles[0]),
				t_emptypoints.size(),
				t_emptytriangles.size(),
				*numberofemptypoints,
				*numberofemptytriangles,
				*numberofsubseg,
				numberofmids,
				encmode,
				theta,
				iteration);

#ifdef GQM2D_DEBUG
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
#endif

			// update neighbors information
			numberofblocks = (ceil)((float)numberofmids / BLOCK_SIZE);
			kernelUpdateMidNeighbors << <numberofblocks, BLOCK_SIZE >> >(
				thrust::raw_pointer_cast(&t_pointlist[0]),
				thrust::raw_pointer_cast(&t_trianglelist[0]),
				thrust::raw_pointer_cast(&t_neighborlist[0]),
				thrust::raw_pointer_cast(&t_tri2subseg[0]),
				thrust::raw_pointer_cast(&t_TStatus[0]),
				thrust::raw_pointer_cast(&t_subseg2tri[0]),
				thrust::raw_pointer_cast(&t_encmarker[0]),
				thrust::raw_pointer_cast(&t_enclist[0]),
				thrust::raw_pointer_cast(&t_internalmarker[0]),
				thrust::raw_pointer_cast(&t_internallist[0]),
				numberofmids,
				encmode,
				theta);

#ifdef GQM2D_DEBUG
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());
#endif
		}

		// Update iteration variables
		// (1) check if there are any slots before last points/triangles
		// (2) update last points/triangles/subsegs
		// (3) update number of empty points/triangles

		int slot_before, slot_after;

		// point slots
		slot_after = t_PStatus.size() - *numberofpoints;
		slot_before = *numberofemptypoints - slot_after;
		if (slot_before < numberofencs)
			*numberofpoints += numberofencs - slot_before;
		*numberofemptypoints -= numberofencs;

		// triangle slots
		slot_after = t_TStatus.size() - *numberoftriangles;
		slot_before = *numberofemptytriangles - slot_after;
		if (slot_before < 2 * numberofencs)
			*numberoftriangles += 2 * numberofencs - slot_before;
		*numberofemptytriangles -= 2 * numberofencs;

		// subseg
		*numberofsubseg += numberofencs;

#ifdef GQM2D_DEBUG_3
		// check if encroachment markers are updated correctly
		{
			printf(" Iteration %d: After Insert mid points\n", iteration);
			int * debug_em = new int[*numberofsubseg];
			int * debug_tl = new int[3 * (*numberoftriangles)];
			int * debug_nl = new int[3 * (*numberoftriangles)];
			REAL2 * debug_pl = new REAL2[*numberofpoints];
			int * debug_st = new int[*numberofsubseg];
			cudaMemcpy(debug_em, thrust::raw_pointer_cast(&t_encmarker[0]), sizeof(int)*(*numberofsubseg), cudaMemcpyDeviceToHost);
			cudaMemcpy(debug_tl, thrust::raw_pointer_cast(&t_trianglelist[0]), sizeof(int) * 3 * *numberoftriangles, cudaMemcpyDeviceToHost);
			cudaMemcpy(debug_nl, thrust::raw_pointer_cast(&t_neighborlist[0]), sizeof(int) * 3 * *numberoftriangles, cudaMemcpyDeviceToHost);
			cudaMemcpy(debug_pl, thrust::raw_pointer_cast(&t_pointlist[0]), sizeof(REAL2)**numberofpoints, cudaMemcpyDeviceToHost);
			cudaMemcpy(debug_st, thrust::raw_pointer_cast(&t_subseg2tri[0]), sizeof(int)**numberofsubseg, cudaMemcpyDeviceToHost);

			for (int i = 0; i < *numberofsubseg; i++)
			{
				int otri = debug_st[i];
				int tri = otri >> 2;
				int ori = otri & 3;
				int p[3];
				REAL2 v[3];
				p[0] = debug_tl[3 * tri + (ori + 1) % 3];
				p[1] = debug_tl[3 * tri + (ori + 2) % 3];
				p[2] = debug_tl[3 * tri + ori];
				v[0] = debug_pl[p[0]];
				v[1] = debug_pl[p[1]];
				v[2] = debug_pl[p[2]];
				bool tag = false; // indicate if this segment is encroached or not

				REAL goodcoss = cos(theta * PI / 180.0);
				goodcoss *= goodcoss;
				REAL dotproduct = (v[0].x - v[2].x)*(v[1].x - v[2].x) +
					(v[0].y - v[2].y)*(v[1].y - v[2].y);

				if (dotproduct < 0.0) // angle > 90
				{
					// here, we use diametral lens to speedup the algorithm
					if (encmode || dotproduct * dotproduct >=
						(2.0*goodcoss - 1.0)*(2.0*goodcoss - 1.0) *
						((v[0].x - v[2].x)*(v[0].x - v[2].x) + (v[0].y - v[2].y)*(v[0].y - v[2].y)) *
						((v[1].x - v[2].x)*(v[1].x - v[2].x) + (v[1].y - v[2].y)*(v[1].y - v[2].y)))
						tag = true;
				}

				otri = debug_nl[3 * tri + ori];
				if (otri != -1)
				{
					tri = otri >> 2;
					ori = otri & 3;
					p[2] = debug_tl[3 * tri + ori];
					v[2] = debug_pl[p[2]];
					dotproduct = (v[0].x - v[2].x)*(v[1].x - v[2].x) +
						(v[0].y - v[2].y)*(v[1].y - v[2].y);
					if (dotproduct < 0.0) // angle > 90
					{
						// here, we use diametral lens to speedup the algorithm
						if (encmode || dotproduct * dotproduct >=
							(2.0*goodcoss - 1.0)*(2.0*goodcoss - 1.0) *
							((v[0].x - v[2].x)*(v[0].x - v[2].x) + (v[0].y - v[2].y)*(v[0].y - v[2].y)) *
							((v[1].x - v[2].x)*(v[1].x - v[2].x) + (v[1].y - v[2].y)*(v[1].y - v[2].y)))
							tag = true;
					}
				}

				if (debug_em[i] == -1 && tag)
					printf(" iteration = %d, Segment %d: I am encroached but marked as non-encroached\n", i, iteration);

				if (debug_em[i] == 0 && !tag)
					printf(" iteration = %d, Segment %d: I am not encroached but marked as encroached\n", i, iteration);
			}
			delete[] debug_em;
			delete[] debug_tl;
			delete[] debug_nl;
			delete[] debug_pl;
			delete[] debug_st;
		}

		// Check if contain duplicate vertices
		{
			int * debug_tl = new int[3 * (*numberoftriangles)];
			REAL2 * debug_pl = new REAL2[*numberofpoints];
			TStatus * debug_ts = new TStatus[*numberoftriangles];
			cudaMemcpy(debug_tl, thrust::raw_pointer_cast(&t_trianglelist[0]), sizeof(int) * 3 * *numberoftriangles, cudaMemcpyDeviceToHost);
			cudaMemcpy(debug_pl, thrust::raw_pointer_cast(&t_pointlist[0]), sizeof(REAL2)**numberofpoints, cudaMemcpyDeviceToHost);
			cudaMemcpy(debug_ts, thrust::raw_pointer_cast(&t_TStatus[0]), sizeof(TStatus)**numberoftriangles, cudaMemcpyDeviceToHost);
			for (int i = 0; i < *numberoftriangles; i++)
			{
				if (!debug_ts[i].isNull())
				{
					bool errorflag = false;
					int p[3];
					REAL2 v[3];
					for (int j = 0; j<3; j++)
					{
						p[j] = debug_tl[3 * i + j];
						v[j] = debug_pl[p[j]];
					}
					for (int j = 0; j<2; j++)
					{
						for (int k = j + 1; k<3; k++)
						{
							if (v[j].x == v[k].x && v[j].y == v[k].y)
							{
								errorflag = true;
							}
						}
					}
					if (errorflag)
						printf(" After insert midpoints - Tri %d (%d, %d, %d): Duplicate vertice\n", i, p[0], p[1], p[2]);
				}
			}
			delete[] debug_tl;
			delete[] debug_pl;
			delete[] debug_ts;
		}
#endif

		// maintain denauly property, do flip-flop
		flipFlop(
			t_pointlist,
			t_PStatus,
			t_trianglelist,
			t_neighborlist,
			t_tri2subseg,
			t_TStatus,
			t_subseg2tri,
			t_flipBy,
			t_flipActive,
			t_encmarker,
			t_linklist,
			t_linkslot,
			*numberoftriangles,
			encmode,
			theta,
			-1,
			-1);

		// check if encroachment markers are updated correctly
#ifdef GQM2D_DEBUG_3
		{
			printf(" Iteration %d: After Insert midpoints and flipFlop\n", iteration);
			int * debug_em = new int[*numberofsubseg];
			int * debug_tl = new int[3 * (*numberoftriangles)];
			int * debug_nl = new int[3 * (*numberoftriangles)];
			REAL2 * debug_pl = new REAL2[*numberofpoints];
			int * debug_st = new int[*numberofsubseg];
			cudaMemcpy(debug_em, thrust::raw_pointer_cast(&t_encmarker[0]), sizeof(int)*(*numberofsubseg), cudaMemcpyDeviceToHost);
			cudaMemcpy(debug_tl, thrust::raw_pointer_cast(&t_trianglelist[0]), sizeof(int) * 3 * *numberoftriangles, cudaMemcpyDeviceToHost);
			cudaMemcpy(debug_nl, thrust::raw_pointer_cast(&t_neighborlist[0]), sizeof(int) * 3 * *numberoftriangles, cudaMemcpyDeviceToHost);
			cudaMemcpy(debug_pl, thrust::raw_pointer_cast(&t_pointlist[0]), sizeof(REAL2)**numberofpoints, cudaMemcpyDeviceToHost);
			cudaMemcpy(debug_st, thrust::raw_pointer_cast(&t_subseg2tri[0]), sizeof(int)**numberofsubseg, cudaMemcpyDeviceToHost);

			for (int i = 0; i< *numberofsubseg; i++)
			{
				int otri = debug_st[i];
				int tri = otri >> 2;
				int ori = otri & 3;
				int p[3];
				REAL2 v[3];
				p[0] = debug_tl[3 * tri + (ori + 1) % 3];
				p[1] = debug_tl[3 * tri + (ori + 2) % 3];
				p[2] = debug_tl[3 * tri + ori];
				v[0] = debug_pl[p[0]];
				v[1] = debug_pl[p[1]];
				v[2] = debug_pl[p[2]];
				bool tag = false; // indicate if this segment is encroached or not

				REAL goodcoss = cos(theta * PI / 180.0);
				goodcoss *= goodcoss;
				REAL dotproduct = (v[0].x - v[2].x)*(v[1].x - v[2].x) +
					(v[0].y - v[2].y)*(v[1].y - v[2].y);

				if (dotproduct < 0.0) // angle > 90
				{
					// here, we use diametral lens to speedup the algorithm
					if (encmode || dotproduct * dotproduct >=
						(2.0*goodcoss - 1.0)*(2.0*goodcoss - 1.0) *
						((v[0].x - v[2].x)*(v[0].x - v[2].x) + (v[0].y - v[2].y)*(v[0].y - v[2].y)) *
						((v[1].x - v[2].x)*(v[1].x - v[2].x) + (v[1].y - v[2].y)*(v[1].y - v[2].y)))
						tag = true;
				}

				otri = debug_nl[3 * tri + ori];
				if (otri != -1)
				{
					tri = otri >> 2;
					ori = otri & 3;
					p[2] = debug_tl[3 * tri + ori];
					v[2] = debug_pl[p[2]];
					dotproduct = (v[0].x - v[2].x)*(v[1].x - v[2].x) +
						(v[0].y - v[2].y)*(v[1].y - v[2].y);
					if (dotproduct < 0.0) // angle > 90
					{
						// here, we use diametral lens to speedup the algorithm
						if (encmode || dotproduct * dotproduct >=
							(2.0*goodcoss - 1.0)*(2.0*goodcoss - 1.0) *
							((v[0].x - v[2].x)*(v[0].x - v[2].x) + (v[0].y - v[2].y)*(v[0].y - v[2].y)) *
							((v[1].x - v[2].x)*(v[1].x - v[2].x) + (v[1].y - v[2].y)*(v[1].y - v[2].y)))
							tag = true;
					}
				}

				if (debug_em[i] == -1 && tag)
					printf(" iteration = %d, Segment %d: I am encroached but marked as non-encroached\n", i, iteration);

				if (debug_em[i] == 0 && !tag)
					printf(" iteration = %d, Segment %d: I am not encroached but marked as encroached\n", i, iteration);

			}
			printf("Finished Checking\n");
		}
#endif

		iteration++;
	}
	//printf("splitEncsegs - totally %d iterations\n",iteration);
}